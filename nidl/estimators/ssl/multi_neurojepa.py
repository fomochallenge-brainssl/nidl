from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any, Optional, Union

import torch
import torch.nn.functional as F  # noqa: N812
from torch import nn
from torch.optim import Optimizer

from nidl.volume.backbones.vit3d_moe import (
    Block,
    apply_masks,
    moe_bias_update,
    trunc_normal_,
)
from nidl.estimators.base import BaseEstimator, TransformerMixin
from nidl.estimators.ssl.utils.encoder import build_encoder
from nidl.estimators.ssl.utils.momentum import (
    MomentumUpdater,
    initialize_momentum_params,
)
from nidl.estimators.ssl.utils.optimizer import configure_ssl_optimizers
from nidl.utils.data_parsing import parse_x_or_xy_batch

from .neurojepa import (
    DEFAULT_MULTISCALE_MASK_CONFIG,
    MaskScaleConfig,
    MultiScaleMaskCollator,
    compute_foreground_patches,
)

# --------------------------------------------------------------------------
# Design summary
# --------------------------------------------------------------------------
# MultiMAE's recipe, adapted to NeuroJEPA's index-driven (rotary-style) 3D
# ViT: keep ONE shared `Block` stack (the `blocks` you already have), and
# give every modality its own *input adapter* (a `Conv3d` patch embed +
# a learned modality-embedding vector) so the shared blocks receive a
# single token sequence that mixes patches from every modality. Since
# `Block.forward(x, mask=pos_idx, D_patches=, H_patches=, W_patches=)`
# derives its 3D positional/relative encoding purely from the *value* of
# each token's `pos_idx` (its flat index into the shared (H, W, D) patch
# grid -- see how NeuroJEPA's own predictor builds `pos_idx` by
# concatenating context/target index tensors), two tokens from different
# modalities at the *same* spatial patch can legitimately share the same
# `pos_idx`: that's not a collision, it's the correct encoding of
# "co-located, different channel" (zero relative spatial distance).
# Modality identity is instead carried in the token *content*, via the
# additive modality-embedding baked in at the patch-embed stage (encoder
# side) and re-derived for mask tokens via a `modality_embed` table
# (predictor side).
#
# Masking has two modes, both built on top of the *existing*,
# untouched `_MaskGenerator` / `MultiScaleMaskCollator`:
#   - "independent": one `MultiScaleMaskCollator` per modality, each
#     drawing its own mask geometry (own RNG stream, see
#     `MultiModalMaskCollator._mask_one`'s explicit per-modality seed).
#   - "shared": a single `MultiScaleMaskCollator` draws ONE (enc, pred)
#     split, reused verbatim (same indices) for every modality, so the
#     same spatial region is masked everywhere.
# --------------------------------------------------------------------------


class MultiModalPatchEmbed3D(nn.Module):
    """Per-modality Conv3d patch embedding + learned modality embedding.

    One `Conv3d(in_channels, embed_dim, kernel_size=patch_size,
    stride=patch_size)` per modality (so intensity statistics of T1w vs.
    DWI etc. get their own linear read-in), followed by adding a
    per-modality embedding vector so the (shared) transformer blocks can
    tell tokens from different modalities apart even when they share the
    same spatial `pos_idx`.

    Parameters
    ----------
    num_modalities : int
    in_channels : int
        Channels per modality volume (typically 1 for MRI intensity maps).
    patch_size : tuple[int, int, int]
    embed_dim : int
    grid_shape : tuple[int, int, int]
        (nH, nW, nD); only used to sanity-check inputs at forward time.
    init_std : float
    """

    def __init__(
        self,
        num_modalities: int,
        in_channels: int,
        patch_size: tuple[int, int, int],
        embed_dim: int,
        grid_shape: tuple[int, int, int],
        init_std: float = 0.02,
    ):
        super().__init__()
        self.num_modalities = num_modalities
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.grid_shape = grid_shape

        self.proj = nn.ModuleList(
            [
                nn.Conv3d(
                    in_channels,
                    embed_dim,
                    kernel_size=patch_size,
                    stride=patch_size,
                )
                for _ in range(num_modalities)
            ]
        )
        self.modality_embed = nn.Parameter(
            torch.zeros(num_modalities, 1, embed_dim)
        )

        for conv in self.proj:
            # standard ViT patch-embed init: treat the flattened kernel as
            # a linear layer's weight.
            w = conv.weight.data
            trunc_normal_(w.view(w.shape[0], -1), std=init_std)
            nn.init.zeros_(conv.bias)
        trunc_normal_(self.modality_embed, std=init_std)

    def forward(self, x: torch.Tensor, modality_index: int) -> torch.Tensor:
        """x : (B, C, H, W, D) -> (B, N, embed_dim), N = prod(grid_shape)."""
        tok = self.proj[modality_index](x)  # (B, E, nH, nW, nD)
        tok = tok.flatten(2).transpose(1, 2)  # (B, N, E)
        return tok + self.modality_embed[modality_index]


class MultiModalNeuroJEPAEncoderWrapper(nn.Module):
    """Endows a NeuroJEPA-style 3D ViT's *shared blocks* with per-modality
    patch embeddings, so ONE `Block` stack jointly processes masked tokens
    gathered from several co-registered modalities.

    Deliberately does NOT reuse `vit.forward` (which does its own
    single-modality patch-embed -> blocks pipeline): only `vit.blocks` is
    shared. Patch embedding is entirely handled by
    `MultiModalPatchEmbed3D`, owned by this wrapper, so it gets its own
    (trainable, EMA-tracked like everything else) parameters.

    Parameters
    ----------
    vit : nn.Module
        Same NeuroJEPA 3D-ViT interface as `NeuroJEPAEncoderWrapper`
        (must expose `embed_dim`, `patch_size`, `grid_shape`, `blocks`).
        Only `embed_dim` / `patch_size` / `grid_shape` / `blocks` are used;
        `vit.forward` itself is never called.
    num_modalities : int
    in_channels : int, default=1
    init_std : float, default=0.02

    Raises
    ------
    TypeError
        If ``vit`` is missing any of the required attributes.
    """

    _REQUIRED = ("embed_dim", "patch_size", "grid_shape", "blocks")

    def __init__(
        self,
        vit: nn.Module,
        num_modalities: int,
        in_channels: int = 1,
        init_std: float = 0.02,
    ):
        super().__init__()
        missing = [a for a in self._REQUIRED if not hasattr(vit, a)]
        if missing:
            raise TypeError(
                "encoder must follow the NeuroJEPA 3D-ViT interface, "
                f"missing {missing}."
            )
        self.vit = vit
        self.num_modalities = num_modalities
        self.patch_embed = MultiModalPatchEmbed3D(
            num_modalities=num_modalities,
            in_channels=in_channels,
            patch_size=vit.patch_size,
            embed_dim=vit.embed_dim,
            grid_shape=vit.grid_shape,
            init_std=init_std,
        )

    @property
    def embed_dim(self) -> int:
        return self.vit.embed_dim

    @property
    def patch_size(self) -> tuple[int, int, int]:
        return self.vit.patch_size

    @property
    def grid_shape(self) -> tuple[int, int, int]:
        return self.vit.grid_shape

    @property
    def num_patches(self) -> int:
        h, w, d = self.grid_shape
        return h * w * d

    @property
    def blocks(self) -> nn.ModuleList:
        return self.vit.blocks

    def forward(
        self,
        xs: list[torch.Tensor],
        masks: Optional[list[list[torch.Tensor]]] = None,
    ):
        """
        Parameters
        ----------
        xs : list of length M (num_modalities)
            Each entry (B, C, H, W, D), all sharing `grid_shape`.
        masks : None, or list of length S ("scales")
            Each entry is itself a list of length M of (B, K_{s,m})
            LongTensors -- the patch-grid indices to keep for modality m
            at scale s (mirrors `NeuroJEPAEncoderWrapper`'s `masks`
            argument, generalized with an inner per-modality axis). If
            None, every modality is fully embedded and concatenated
            (used by `forward_target`, mirroring `forward_target`'s
            `masks=None` full pass in the base `NeuroJEPA`).

        Returns
        -------
        tokens : torch.Tensor
            (B*S, sum_m K_{s,m}, E) if `masks` given (concatenated,
            scale-major then modality-major -- matches
            `MultiModalVisionTransformerPredictor3D`'s expected layout),
            else (B, M*N, E) with modality-major layout (segment m is
            `tokens[:, m*N:(m+1)*N, :]`).
        moe_scores : list
            Non-None per-block MoE routing scores, if any.
        """
        M = len(xs)
        if self.num_modalities != M:
            raise ValueError(
                f"expected {self.num_modalities} modalities, got {M}"
            )
        B = xs[0].shape[0]
        N = self.num_patches
        grid_h, grid_w, grid_d = self.grid_shape

        tokens_full = [
            self.patch_embed(xs[m], m) for m in range(M)
        ]  # list[M] of (B, N, E)

        if masks is None:
            x = torch.cat(tokens_full, dim=1)  # (B, M*N, E)
            pos_idx = (
                torch.arange(N, device=x.device)
                .repeat(M)
                .unsqueeze(0)
                .expand(B, -1)
            )
        else:
            x_list, pos_list = [], []
            for scale_masks in masks:
                if len(scale_masks) != M:
                    raise ValueError(
                        "each `masks` entry must have one index tensor "
                        f"per modality ({M}), got {len(scale_masks)}"
                    )
                toks = [
                    apply_masks(
                        tokens_full[m], [scale_masks[m]], concat=False
                    )[0]
                    for m in range(M)
                ]
                x_list.append(torch.cat(toks, dim=1))
                pos_list.append(torch.cat(scale_masks, dim=1))
            x = torch.cat(x_list, dim=0)
            pos_idx = torch.cat(pos_list, dim=0)

        moe_scores = []
        for blk in self.blocks:
            x, score = blk(
                x,
                mask=pos_idx,
                D_patches=grid_d,
                H_patches=grid_h,
                W_patches=grid_w,
            )
            if score is not None:
                moe_scores.append(score)
        return x, moe_scores


class MultiModalVisionTransformerPredictor3D(nn.Module):
    """Multimodal counterpart of `VisionTransformerPredictor3D`.

    Same idea (context tokens in, per-target-position predictions out,
    shared `Block` stack), extended with a `modality_embed` table so a
    mask token can be told "predict modality m's patch at this spatial
    location" -- the spatial location itself is still communicated purely
    through `pos_idx`, exactly as in the single-modality version.

    Unlike `VisionTransformerPredictor3D`, this does not implement the
    general cross-product-of-scales machinery (`repeat_interleave_batch`):
    `MultiNeuroJEPA._shared_step`, like `NeuroJEPA._shared_step`, always
    calls this predictor once per mask scale with single-element
    `masks_x`/`masks_y` lists, so that generality is dropped in favor of a
    simpler implementation (enforced by an assertion below).
    """

    def __init__(
        self,
        grid_shape: tuple[int, int, int],
        num_modalities: int,
        embed_dim: int = 768,
        predictor_embed_dim: int = 384,
        depth: int = 6,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        num_mask_tokens: int = 1,
        init_std: float = 0.02,
    ):
        super().__init__()
        self.grid_h, self.grid_w, self.grid_d = grid_shape
        self.num_patches = self.grid_h * self.grid_w * self.grid_d
        self.num_modalities = num_modalities

        self.predictor_embed = nn.Linear(
            embed_dim, predictor_embed_dim, bias=True
        )
        self.num_mask_tokens = num_mask_tokens
        self.mask_tokens = nn.ParameterList(
            [
                nn.Parameter(torch.zeros(1, 1, predictor_embed_dim))
                for _ in range(num_mask_tokens)
            ]
        )
        self.modality_embed = nn.Parameter(
            torch.zeros(num_modalities, 1, 1, predictor_embed_dim)
        )

        self.predictor_blocks = nn.ModuleList(
            [
                Block(
                    dim=predictor_embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    grid_size=self.grid_h,
                    use_moe=False,
                )
                for _ in range(depth)
            ]
        )
        self.predictor_norm = nn.LayerNorm(predictor_embed_dim)
        self.predictor_proj = nn.Linear(
            predictor_embed_dim, embed_dim, bias=True
        )

        self.init_std = init_std
        for mt in self.mask_tokens:
            trunc_normal_(mt, std=init_std)
        trunc_normal_(self.modality_embed, std=init_std)
        self.apply(self._init_weights)
        for layer_id, blk in enumerate(self.predictor_blocks):
            blk.attn.proj.weight.data.div_(math.sqrt(2.0 * (layer_id + 1)))
            blk.mlp.fc2.weight.data.div_(math.sqrt(2.0 * (layer_id + 1)))

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=self.init_std)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(
        self,
        x: torch.Tensor,
        masks_x: list[list[torch.Tensor]],
        masks_y: list[list[torch.Tensor]],
        mask_index: int = 0,
    ):
        """
        x : (B, K_ctx_total, E) -- output of
            `MultiModalNeuroJEPAEncoderWrapper` for ONE scale.
        masks_x, masks_y : single-element lists ([scale_masks]), each
            `scale_masks` a list of M (B, K_m) index tensors -- the exact
            same objects passed as `masks=[scale_masks]` to the context
            encoder for this scale (masks_x) and the matching target split
            (masks_y).

        Returns
        -------
        (B, sum_m K_pred_m, E), modality-major (same ordering
        `MultiNeuroJEPA.forward_target` uses for the EMA target side).
        """
        if len(masks_x) != 1 or len(masks_y) != 1:
            raise ValueError(
                "MultiModalVisionTransformerPredictor3D only supports one "
                "mask scale per call (MultiNeuroJEPA._shared_step calls it "
                "once per scale, matching NeuroJEPA's convention)."
            )
        mx = masks_x[0]
        my = masks_y[0]
        M = self.num_modalities
        B = mx[0].shape[0]

        x = self.predictor_embed(x)
        N_ctxt = x.shape[1]

        mask_index = mask_index % self.num_mask_tokens
        pred_segments, pos_segments = [], []
        for m in range(M):
            idx = my[m]  # (B, K_m)
            tok = self.mask_tokens[mask_index] + self.modality_embed[m]
            tok = tok.expand(B, idx.shape[1], -1)
            pred_segments.append(tok)
            pos_segments.append(idx)
        pred_tok = torch.cat(pred_segments, dim=1)
        pred_pos = torch.cat(pos_segments, dim=1)
        ctx_pos = torch.cat(mx, dim=1)

        x = torch.cat([x, pred_tok], dim=1)
        pos_idx = torch.cat([ctx_pos, pred_pos], dim=1)

        for blk in self.predictor_blocks:
            x, _ = blk(
                x,
                mask=pos_idx,
                D_patches=self.grid_d,
                H_patches=self.grid_h,
                W_patches=self.grid_w,
            )
        x = self.predictor_norm(x)
        x = x[:, N_ctxt:]
        return self.predictor_proj(x)


class MultiModalMaskCollator:
    """Produces per-modality (context, target) index splits for
    `MultiNeuroJEPA`, in one of two strategies, both built strictly on top
    of the *unmodified* `_MaskGenerator` (accessed via a plain
    `MultiScaleMaskCollator`'s `.generators`).

    Parameters
    ----------
    grid_shape, patch_size, scale_configs, foreground_threshold,
    min_foreground_fraction : same meaning as `MultiScaleMaskCollator`.
    num_modalities : int
    strategy : {"independent", "shared"}
        - "independent": each modality draws its own mask geometry per
          scale (separate RNG stream per modality, see `seed` below).
        - "shared": ONE mask geometry per scale is drawn and reused,
          index-for-index, across every modality.
    foreground_aware : bool, default=True
        Per-modality foreground maps are always computed when True (used
        for the loss regardless of strategy). For "shared" masking, the
        per-modality maps are combined with a logical OR (a patch counts
        as foreground for mask-erosion purposes if it is foreground in
        *any* modality) before being handed to the single shared
        `_MaskGenerator` call.
    """

    def __init__(
        self,
        grid_shape: tuple[int, int, int],
        patch_size: tuple[int, int, int],
        num_modalities: int,
        scale_configs: Sequence[
            MaskScaleConfig
        ] = DEFAULT_MULTISCALE_MASK_CONFIG,
        strategy: str = "independent",
        foreground_aware: bool = True,
        foreground_threshold: float = 0.0,
        min_foreground_fraction: float = 0.1,
        modality_seed_multiplier: int = 999_331,
    ):
        if strategy not in ("independent", "shared"):
            raise ValueError(
                f"strategy must be 'independent' or 'shared', got {strategy!r}"
            )
        self.strategy = strategy
        self.num_modalities = num_modalities
        self.patch_size = patch_size
        self.foreground_aware = foreground_aware
        self.foreground_threshold = foreground_threshold
        self.min_foreground_fraction = min_foreground_fraction
        self.modality_seed_multiplier = modality_seed_multiplier

        n_collators = num_modalities if strategy == "independent" else 1
        # Only used for their `.generators` (a list of `_MaskGenerator`,
        # one per scale config); their own internal seeding is bypassed in
        # favor of the explicit per-modality seeding below, to guarantee
        # different modalities draw statistically independent masks even
        # when using identical `MaskScaleConfig`s at the same step.
        self._collators = [
            MultiScaleMaskCollator(
                grid_shape=grid_shape,
                patch_size=patch_size,
                scale_configs=scale_configs,
            )
            for _ in range(n_collators)
        ]

        self._step_counter = -1
        self._rank_seed_offset = 0

    def set_rank(self, rank: int, large_multiplier: int = 10_000_000) -> None:
        """Call once per DDP process, exactly like
        `NeuroJEPA.on_fit_start` does for `MultiScaleMaskCollator`."""
        self._rank_seed_offset = rank * large_multiplier

    def step(self) -> int:
        self._step_counter += 1
        return self._step_counter + self._rank_seed_offset

    def _mask_one(
        self, collator: MultiScaleMaskCollator, batch_size, seed, fg
    ):
        masks_enc, masks_pred = [], []
        for gen in collator.generators:
            enc, pred = gen(batch_size, seed, foreground_mask=fg)
            masks_enc.append(enc)
            masks_pred.append(pred)
        return masks_enc, masks_pred

    def __call__(self, volumes: list[torch.Tensor]):
        """
        volumes : list of length M, each (B, C, H, W, D).

        Returns
        -------
        masks_enc, masks_pred : list of length M, each a list of length S
            (num scales) of (B, K) LongTensors.
        fg_flat : list of length M of (B, N) tensors, or [None]*M if
            `foreground_aware=False`.
        """
        M = self.num_modalities
        if len(volumes) != M:
            raise ValueError(f"expected {M} modalities, got {len(volumes)}")
        B = volumes[0].shape[0]

        fg_per_mod = None
        if self.foreground_aware:
            fg_per_mod = [
                compute_foreground_patches(
                    v,
                    self.patch_size,
                    self.foreground_threshold,
                    self.min_foreground_fraction,
                ).cpu()
                for v in volumes
            ]

        base_seed = self.step()

        if self.strategy == "independent":
            masks_enc, masks_pred = [], []
            for m in range(M):
                seed_m = base_seed * self.modality_seed_multiplier + m
                fg_m = fg_per_mod[m] if fg_per_mod is not None else None
                enc, pred = self._mask_one(self._collators[m], B, seed_m, fg_m)
                masks_enc.append(enc)
                masks_pred.append(pred)
        else:
            fg_combined = None
            if fg_per_mod is not None:
                fg_combined = torch.stack(fg_per_mod, dim=0).any(dim=0)
            seed0 = base_seed * self.modality_seed_multiplier
            enc, pred = self._mask_one(
                self._collators[0], B, seed0, fg_combined
            )
            masks_enc = [enc for _ in range(M)]
            masks_pred = [pred for _ in range(M)]

        fg_flat = [
            f.flatten(1).float() if f is not None else None
            for f in (fg_per_mod if fg_per_mod is not None else [None] * M)
        ]
        return masks_enc, masks_pred, fg_flat


def multimodal_foreground_aware_jepa_loss(
    z: list[torch.Tensor],
    h: list[torch.Tensor],
    masks_pred: list[list[torch.Tensor]],
    loss_exp: float = 1.0,
    fg_flat: Optional[list[Optional[torch.Tensor]]] = None,
    bg_weight: float = 1.0,
) -> torch.Tensor:
    """Multimodal counterpart of `foreground_aware_jepa_loss`.

    z, h : list of length S (num scales), each (B, sum_m K_{s,m}, E),
        modality-major concatenation (as produced by
        `MultiModalVisionTransformerPredictor3D` / `forward_target`).
    masks_pred : list of length M of list of length S of (B, K_{s,m})
        index tensors -- used to know each modality's segment boundaries
        within z[s]/h[s], and to gather the matching foreground weights.
    fg_flat : list of length M of (B, N) tensors, or None entries -- as
        returned by `MultiModalMaskCollator`.
    """
    use_weighting = (
        fg_flat is not None
        and bg_weight < 1.0
        and all(f is not None for f in fg_flat)
    )
    M = len(masks_pred)
    S = len(masks_pred[0])

    loss, n = 0.0, 0
    for s in range(S):
        zi, hi = z[s], h[s]
        per_token = torch.abs(zi - hi) ** loss_exp / loss_exp
        per_token = per_token.mean(dim=-1)  # (B, K_total_s)

        if use_weighting:
            fg_segments = []
            for m in range(M):
                idx = masks_pred[m][s]
                seg = apply_masks(
                    fg_flat[m].unsqueeze(-1), [idx], concat=False
                )[0].squeeze(-1)
                fg_segments.append(seg)
            fg_vals = torch.cat(fg_segments, dim=1)
            w = bg_weight + (1.0 - bg_weight) * fg_vals
            w = w / w.mean(dim=-1, keepdim=True).clamp(min=1e-6)
            loss = loss + (per_token * w).mean()
        else:
            loss = loss + per_token.mean()
        n += 1
    return loss / n


def parse_multimodal_batch(
    batch: Any,
    modalities: Sequence[str],
    device: Optional[torch.device] = None,
) -> list[torch.Tensor]:
    """Splits off any label via `parse_x_or_xy_batch`, then normalizes the
    remaining `X` into a fixed-order list of per-modality tensors.

    Accepts `X` as either a dict `{modality_name: (B,C,H,W,D) tensor}`
    (keyed by entries of `modalities`) or a plain `(x_1, ..., x_M)`
    tuple/list already in `modalities` order.
    """
    x, _ = parse_x_or_xy_batch(batch, device=device)
    if isinstance(x, dict):
        xs = [x[m] for m in modalities]
    elif isinstance(x, (list, tuple)):
        xs = list(x)
    else:
        raise TypeError(
            "MultiNeuroJEPA expects the batch's X to be a dict "
            "{modality_name: (B,C,H,W,D) tensor} keyed by `self.modalities`,"
            " or a (x_mod_1, ..., x_mod_M) tuple/list in that same order; "
            f"got {type(x)}."
        )
    if len(xs) != len(modalities):
        raise ValueError(
            f"expected {len(modalities)} modalities ({list(modalities)}), "
            f"got {len(xs)}"
        )
    if device is not None:
        xs = [xi.to(device, non_blocking=True) for xi in xs]
    return xs


class MultiNeuroJEPA(TransformerMixin, BaseEstimator):
    """Multimodal extension of `NeuroJEPA` (MultiMAE-style): one shared 3D
    ViT (`context_encoder`/`target_encoder` blocks) processes masked
    tokens gathered from several co-registered modalities (e.g. T1w, T2w,
    DWI, FLAIR -- any set sharing the same 3D shape and `grid_shape`).

    Only the masking / tokenization differs from `NeuroJEPA`; the
    training loop shape (multi-scale block masking, EMA target encoder,
    predictor, foreground-aware loss, optional MoE bias update) is
    otherwise identical, one scale at a time, summed across scales.

    Parameters
    ----------
    encoder : nn.Module
        Single-modality 3D ViT-like module (same contract as `NeuroJEPA`'s
        `encoder`): exposes `embed_dim`, `patch_size`, `grid_shape`,
        `blocks`. Only its `blocks` (and sizing attributes) are reused;
        `encoder.forward` itself is never called -- patch embedding is
        replaced by per-modality adapters (see
        `MultiModalNeuroJEPAEncoderWrapper`).
    modalities : sequence of str, default=("t1w", "t2w", "dwi", "flair")
        Names and canonical order of the modalities. Determines the
        expected key order for dict-valued batches (see
        `parse_multimodal_batch`).
    mask_strategy : {"independent", "shared"}, default="independent"
        - "independent": each modality is masked with its own draw of
          NeuroJEPA's multi-scale block-masking strategy.
        - "shared": a single mask is drawn per scale and applied
          identically (same patch indices) to every modality.
    in_channels : int, default=1
        Channels per modality volume.
    mask_scale_configs, foreground_aware, foreground_threshold,
    min_foreground_fraction, loss_exp, bg_weight, use_moe,
    moe_bias_update_rate, moe_bias_clip, predictor_embed_dim,
    predictor_depth, predictor_num_heads, optimizer, learning_rate,
    weight_decay, exclude_bias_and_norm_wd, ema_start, ema_end,
    optimizer_kwargs, lr_scheduler, lr_scheduler_kwargs : same meaning as
        `NeuroJEPA`.
    **kwargs : passed to `BaseEstimator`.

    Attributes
    ----------
    context_encoder, target_encoder : MultiModalNeuroJEPAEncoderWrapper
    predictor : MultiModalVisionTransformerPredictor3D
    masker : MultiModalMaskCollator
    """

    def __init__(
        self,
        encoder: nn.Module,
        modalities: Sequence[str] = ("t1w", "t2w", "dwi", "flair"),
        mask_strategy: str = "independent",
        in_channels: int = 1,
        mask_scale_configs: Sequence[
            MaskScaleConfig
        ] = DEFAULT_MULTISCALE_MASK_CONFIG,
        foreground_aware: bool = True,
        foreground_threshold: float = 0.0,
        min_foreground_fraction: float = 0.1,
        loss_exp: float = 1.0,
        bg_weight: float = 0.1,
        use_moe: bool = False,
        moe_bias_update_rate: float = 1e-4,
        moe_bias_clip: float = 0.3,
        predictor_embed_dim: int = 384,
        predictor_depth: int = 6,
        predictor_num_heads: int = 12,
        optimizer: Union[str, Optimizer] = "adamW",
        learning_rate: float = 6e-4,
        weight_decay: float = 0.04,
        exclude_bias_and_norm_wd: bool = True,
        ema_start: float = 0.99925,
        ema_end: float = 1.0,
        optimizer_kwargs: Optional[dict] = None,
        lr_scheduler: Optional[Union[str, Any]] = "warmup_cosine",
        lr_scheduler_kwargs: Optional[dict] = None,
        **kwargs: Any,
    ):
        if mask_strategy not in ("independent", "shared"):
            raise ValueError(
                "mask_strategy must be 'independent' or 'shared', got "
                f"{mask_strategy!r}"
            )

        ignore = kwargs.pop("ignore", ["callbacks", "encoder"])
        if "callbacks" not in ignore:
            ignore.append("callbacks")
        if "encoder" not in ignore:
            ignore.append("encoder")

        super().__init__(**kwargs, ignore=ignore)

        self.modalities = tuple(modalities)
        self.num_modalities = len(self.modalities)
        self.mask_strategy = mask_strategy

        self.target_encoder = MultiModalNeuroJEPAEncoderWrapper(
            build_encoder(encoder, deepcopy=True),
            num_modalities=self.num_modalities,
            in_channels=in_channels,
        )
        self.context_encoder = MultiModalNeuroJEPAEncoderWrapper(
            build_encoder(encoder),
            num_modalities=self.num_modalities,
            in_channels=in_channels,
        )
        initialize_momentum_params(self.context_encoder, self.target_encoder)
        self.momentum_updater = MomentumUpdater(ema_start, ema_end)

        grid_shape = self.target_encoder.grid_shape
        embed_dim = self.target_encoder.embed_dim
        patch_size = self.target_encoder.patch_size

        self.predictor = MultiModalVisionTransformerPredictor3D(
            grid_shape=grid_shape,
            num_modalities=self.num_modalities,
            embed_dim=embed_dim,
            predictor_embed_dim=predictor_embed_dim,
            depth=predictor_depth,
            num_heads=predictor_num_heads,
        )

        self.masker = MultiModalMaskCollator(
            grid_shape=grid_shape,
            patch_size=patch_size,
            num_modalities=self.num_modalities,
            scale_configs=mask_scale_configs,
            strategy=mask_strategy,
            foreground_aware=foreground_aware,
            foreground_threshold=foreground_threshold,
            min_foreground_fraction=min_foreground_fraction,
        )

        self.foreground_aware = foreground_aware
        self.loss_exp = loss_exp
        self.bg_weight = bg_weight
        self.use_moe = use_moe
        self.moe_bias_update_rate = moe_bias_update_rate
        self.moe_bias_clip = moe_bias_clip

        self.optimizer = optimizer
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.exclude_bias_and_norm_wd = exclude_bias_and_norm_wd
        self.lr_scheduler = lr_scheduler
        self.lr_scheduler_kwargs = (
            lr_scheduler_kwargs if lr_scheduler_kwargs is not None else {}
        )
        self.optimizer_kwargs = optimizer_kwargs

        self._fill_default_lr_scheduler_kwargs()

    def _shared_step(self, batch):
        xs = parse_multimodal_batch(batch, self.modalities, device=self.device)

        masks_enc, masks_pred, fg_flat = self.masker(xs)
        masks_enc = [
            [t.to(self.device, non_blocking=True) for t in per_scale]
            for per_scale in masks_enc
        ]
        masks_pred = [
            [t.to(self.device, non_blocking=True) for t in per_scale]
            for per_scale in masks_pred
        ]
        fg_flat = [
            f.to(self.device, non_blocking=True) if f is not None else None
            for f in fg_flat
        ]

        num_scales = len(masks_enc[0])
        M = self.num_modalities

        h_scales = self.forward_target(xs, masks_pred)

        z_scales = []
        for s in range(num_scales):
            scale_masks_enc = [masks_enc[m][s] for m in range(M)]
            scale_masks_pred = [masks_pred[m][s] for m in range(M)]
            zi = self.context_encoder(xs, masks=[scale_masks_enc])[0]
            zi = self.predictor(
                zi, masks_x=[scale_masks_enc], masks_y=[scale_masks_pred]
            )
            z_scales.append(zi)

        loss = multimodal_foreground_aware_jepa_loss(
            z_scales,
            h_scales,
            masks_pred,
            loss_exp=self.loss_exp,
            fg_flat=fg_flat,
            bg_weight=self.bg_weight,
        )
        return {
            "loss": loss,
            "z_pred": z_scales,
            "z_target": h_scales,
        }

    def training_step(self, batch, batch_idx: int):
        """Same contract as `NeuroJEPA.training_step`.

        Parameters
        ----------
        batch : dict[str, Tensor] or sequence[Tensor] (or that paired with
            labels, i.e. `(X, Y)`), see `parse_multimodal_batch`.
        batch_idx : int
            Ignored.
        """
        outputs = self._shared_step(batch)
        self.log("loss/train", outputs["loss"], prog_bar=True, sync_dist=True)
        return outputs

    @torch.no_grad()
    def forward_target(
        self, xs: list[torch.Tensor], masks_pred: list[list[torch.Tensor]]
    ) -> list[torch.Tensor]:
        """Full (unmasked) forward through the EMA target encoder over all
        modalities at once, layer-normed, then sliced per scale into
        modality-major concatenated target latents at the masked
        positions. Mirrors `NeuroJEPA.forward_target`."""
        h_full, _ = self.target_encoder(xs, masks=None)  # (B, M*N, E)
        h_full = F.layer_norm(h_full, (h_full.size(-1),))

        N = self.target_encoder.num_patches
        M = self.num_modalities
        S = len(masks_pred[0])

        h_scales = []
        for s in range(S):
            segs = []
            for m in range(M):
                idx = masks_pred[m][s]
                seg = h_full[:, m * N : (m + 1) * N, :]
                segs.append(apply_masks(seg, [idx], concat=False)[0])
            h_scales.append(torch.cat(segs, dim=1))
        return h_scales

    def on_train_batch_end(self, outputs: dict, batch, batch_idx: int):
        """Same as `NeuroJEPA.on_train_batch_end`: MoE bias update (if
        `use_moe`), then EMA momentum update."""
        if self.use_moe:
            minvio, maxvio = moe_bias_update(
                self.context_encoder,
                self.moe_bias_update_rate,
                self.moe_bias_clip,
            )
            self.log("moe/min_violation", minvio)
            self.log("moe/max_violation", maxvio)

        self.momentum_updater.update(self.context_encoder, self.target_encoder)
        self.log("lambda", self.momentum_updater.cur_lambda)
        self.momentum_updater.update_lambda(
            cur_step=self.trainer.global_step,
            max_steps=self.trainer.estimated_stepping_batches,
        )

    def on_fit_start(self):
        """Decorrelate mask geometry across DDP ranks (see
        `NeuroJEPA.on_fit_start` for the rationale)."""
        self.masker.set_rank(self.trainer.global_rank)

    def validation_step(self, batch: Any, batch_idx: int):
        outputs = self._shared_step(batch)
        self.log("loss/val", outputs["loss"], prog_bar=True, sync_dist=True)
        return outputs

    def test_step(self, batch, batch_idx):
        return

    def transform_step(
        self,
        batch: Any,
        batch_idx: int,
        dataloader_idx: Optional[int] = 0,
    ):
        """Encode (no masking, no predictor) and return one fused
        embedding per sample, averaged over every modality's tokens.

        Parameters
        ----------
        batch : dict[str, Tensor] or sequence[Tensor]
            Per-modality volumes, see `parse_multimodal_batch`.

        Returns
        -------
        features : torch.Tensor
            (B, embed_dim), mean over the `M * N` context-encoder tokens.
        """
        xs = parse_multimodal_batch(batch, self.modalities, device=self.device)
        tokens, _ = self.context_encoder(xs, masks=None)
        return tokens.mean(dim=1)

    def configure_optimizers(self):
        params = [
            {"name": "backbone", "params": self.context_encoder.parameters()},
            {"name": "predictor", "params": self.predictor.parameters()},
        ]
        return configure_ssl_optimizers(
            trainer=self.trainer,
            optim_params=params,
            optimizer=self.optimizer,
            optimizer_kwargs=self.optimizer_kwargs,
            learning_rate=self.learning_rate,
            weight_decay=self.weight_decay,
            exclude_bias_and_norm_wd=self.exclude_bias_and_norm_wd,
            lr_scheduler=self.lr_scheduler,
            lr_scheduler_kwargs=self.lr_scheduler_kwargs,
        )

    def _fill_default_lr_scheduler_kwargs(self):
        if self.lr_scheduler_kwargs is None:
            self.lr_scheduler_kwargs = {}
        self.lr_scheduler_kwargs.setdefault("warmup_epochs", 10)
        self.lr_scheduler_kwargs.setdefault("interval", "step")
        self.lr_scheduler_kwargs.setdefault("warmup_start_lr", 1e-6)
        self.lr_scheduler_kwargs.setdefault("min_lr", 0)
