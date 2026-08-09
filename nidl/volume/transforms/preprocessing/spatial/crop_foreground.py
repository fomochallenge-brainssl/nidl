##########################################################################
# NSAp - Copyright (C) CEA, 2025
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################
from __future__ import annotations

from typing import Callable, Union

import numpy as np
import torch

from .....transforms import TypeTransformInput, VolumeTransform


def _is_positive(data: np.ndarray) -> np.ndarray:
    return data > 0


class CropForeground(VolumeTransform):
    """Crop a 3d volume to the bounding box of its foreground.

    The bounding box is the smallest box enclosing every voxel for which
    `select_fn` returns `True` (any channel), expanded by `margin` voxels
    on every side. This is the nidl counterpart of MONAI's
    `CropForegroundd` [1]_, reimplemented without a MONAI dependency.

    It handles a :class:`numpy.ndarray` or :class:`torch.Tensor` as input
    and returns a consistent output (same type). Input shape must be
    :math:`(C, H, W, D)` or :math:`(H, W, D)` (spatial dimensions).

    Parameters
    ----------
    select_fn: Callable, default=lambda x: x > 0
        Function returning a boolean array of the same shape as its
        input, used to select the foreground voxels. Applied to the
        whole volume at once (all channels together). As for the
        `masking_fn` of
        :class:`nidl.volume.transforms.preprocessing.RobustRescaling`,
        it always receives a :class:`numpy.ndarray`, even when the
        input data is a :class:`torch.Tensor`.

    margin: int or tuple[int, int, int], default=0
        Number of voxels added on each side of the bounding box, per
        spatial dimension. If int, the same margin is used for the 3
        dimensions.

    allow_smaller: bool, default=True
        Whether the bounding box (after adding `margin`) is allowed to
        be clipped by the volume edges. If `True` (default), the box
        never extends past the original volume and the output is a pure
        crop (no padding). If `False`, the full `margin` is guaranteed
        even past the original volume edges, padding the output with
        `padding_mode`/`constant_values` where needed.

    padding_mode: str, default="constant"
        Padding mode used when `allow_smaller=False` and the bounding
        box extends past the volume edges. See
        :class:`nidl.volume.transforms.preprocessing.CropOrPad` for the
        accepted values.

    constant_values: float, default=0.0
        Value used to pad when `padding_mode="constant"`.

    kwargs: dict
        Keyword arguments given to :class:`nidl.transforms.Transform`.

    Notes
    -----
    Unlike its MONAI counterpart, if no voxel satisfies `select_fn`
    anywhere in the volume, the input is returned unchanged instead of
    being replaced by a degenerate, all-zero-sized crop.

    References
    ----------
    .. [1] MONAI Consortium, "CropForegroundd",
           https://docs.monai.io/en/stable/transforms.html#cropforegroundd

    Examples
    --------
    >>> import numpy as np
    >>> from nidl.volume.transforms.preprocessing import CropForeground
    >>> volume = np.zeros((1, 10, 10, 10))
    >>> volume[:, 4:6, 4:6, 4:6] = 1.0
    >>> transform = CropForeground(margin=1)
    >>> transform(volume).shape
    (1, 4, 4, 4)
    """

    def __init__(
        self,
        select_fn: Callable = _is_positive,
        margin: Union[int, tuple[int, int, int]] = 0,
        allow_smaller: bool = True,
        padding_mode: str = "constant",
        constant_values: Union[float, tuple[float, float, float]] = 0.0,
        **kwargs,
    ):
        super().__init__(**kwargs)

        if not callable(select_fn):
            raise TypeError(
                f"`select_fn` must be callable, got {type(select_fn)}"
            )
        self.select_fn = select_fn
        self.margin = self._parse_shape(margin, length=3)
        self.allow_smaller = bool(allow_smaller)
        self.padding_mode = padding_mode
        self.constant_values = constant_values

    def _compute_bounding_box(
        self, mask: np.ndarray, spatial_shape: tuple[int, int, int]
    ) -> tuple[list[int], list[int]]:
        """Compute the (possibly out-of-bounds) start/end indices of the
        foreground bounding box, expanded by `self.margin`.

        Parameters
        ----------
        mask: np.ndarray
            Boolean array of shape `spatial_shape`, `True` on foreground
            voxels.
        spatial_shape: tuple[int, int, int]
            Spatial shape of the volume `mask` was computed from.

        Returns
        -------
        box_start, box_end: list[int], list[int]
            Start (inclusive) and end (exclusive) indices of the
            bounding box, per spatial dimension. Both are `None` if the
            mask has no foreground voxel at all.
        """
        # Reducing over the two inner (contiguous) axes is much cheaper
        # than over the outer ones, so the last two projections are
        # derived from a single pass over axis 0 rather than from two
        # more multi-axis reductions.
        reduced = np.any(mask, axis=0)  # (W, D)
        projections = (
            np.any(mask, axis=(1, 2)),
            np.any(reduced, axis=1),
            np.any(reduced, axis=0),
        )

        box_start = []
        box_end = []
        for axis, projection in enumerate(projections):
            indices = np.flatnonzero(projection)
            if indices.size == 0:  # no foreground
                return None, None
            start = int(indices[0]) - self.margin[axis]
            end = int(indices[-1]) + 1 + self.margin[axis]
            if self.allow_smaller:
                start = max(start, 0)
                end = min(end, spatial_shape[axis])
            box_start.append(start)
            box_end.append(end)
        return box_start, box_end

    def apply_transform(self, data: TypeTransformInput) -> TypeTransformInput:
        """Crop the input data to its foreground bounding box.

        Parameters
        ----------
        data: np.ndarray or torch.Tensor
            The input data with shape :math:`(C, H, W, D)` or
            :math:`(H, W, D)`.

        Returns
        -------
        data: np.ndarray or torch.Tensor
            Cropped (and possibly padded) data, with same type as input.
        """
        is_tensor = isinstance(data, torch.Tensor)
        if is_tensor:
            # Zero-copy for cpu tensors, and numpy comparison kernels are
            # noticeably faster than their torch counterparts here.
            dtype, device = data.dtype, data.device
            data = data.detach().cpu().numpy()

        shape = data.shape
        has_channel = len(shape) == 4
        spatial_shape = shape[1:] if has_channel else shape

        mask = np.asarray(self.select_fn(data))
        if mask.shape != shape:
            raise ValueError(
                "`select_fn` must return a boolean array with the same "
                f"shape as its input, got {mask.shape} != {shape}"
            )
        if has_channel:
            mask = np.any(mask, axis=0)

        box_start, box_end = self._compute_bounding_box(mask, spatial_shape)
        if box_start is None:
            # No foreground found: return input unchanged (see Notes).
            if is_tensor:
                return torch.as_tensor(data, dtype=dtype, device=device)
            return data

        crop_slices = tuple(
            slice(max(start, 0), min(end, spatial_shape[dim]))
            for dim, (start, end) in enumerate(zip(box_start, box_end))
        )
        if has_channel:
            crop_slices = (slice(None),) + crop_slices
        cropped = data[crop_slices]

        pad_before = [max(-s, 0) for s in box_start]
        pad_after = [
            max(e - spatial_shape[dim], 0) for dim, e in enumerate(box_end)
        ]
        if any(pad_before) or any(pad_after):
            pad_widths = list(zip(pad_before, pad_after))
            if has_channel:
                pad_widths = [(0, 0)] + pad_widths
            kwargs_pad = {}
            if self.padding_mode == "constant":
                kwargs_pad["constant_values"] = self.constant_values
            # np.pad already returns a new array.
            cropped = np.pad(
                cropped, pad_widths, mode=self.padding_mode, **kwargs_pad
            )
        else:
            # Slicing returns a view: copy so that the output neither
            # aliases nor keeps the whole input volume alive.
            cropped = cropped.copy()

        if is_tensor:
            cropped = torch.as_tensor(cropped, dtype=dtype, device=device)
        return cropped
