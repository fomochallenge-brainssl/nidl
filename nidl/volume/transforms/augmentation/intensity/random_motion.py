from __future__ import annotations

import numbers
from typing import Optional, Union

import numpy as np
import torch
from nibabel.orientations import aff2axcodes
from scipy.linalg import expm, logm
from scipy.ndimage import affine_transform

from .....transforms import TypeTransformInput, VolumeTransform
from ..spatial.random_affine import RandomAffine


class RandomMotion(VolumeTransform):
    """Add a random MRI k-space motion artifact to a 3d volume.
    Implementation V1: too slow for use within pretraining pipeline.

    Following Shaw et al. (2019) [1]_, subject motion during the
    acquisition is simulated in five steps:

    1. Sample a random movement model: :math:`N \\sim
       \\max(1, \\mathcal{P}(\\lambda))` rigid transforms (rotation +
       translation) occurring at sorted random times :math:`t_i \\sim
       \\mathcal{U}(0, 1)`, combined incrementally in log-Euclidean space
       so that the motion accumulates (drifts) instead of teleporting.
    2. De-mean the movements with respect to a signal-weighted average
       pose, :math:`A_{avg} = \\exp(\\sum_i w_i \\log A_i)`, where the
       weight :math:`w_i` is the image-domain signal carried by the
       k-space band acquired at pose :math:`i`.
    3. Apply each de-meaned transform to the clean volume and resample
       (b-spline interpolation).
    4. Build a composite k-space: each phase-encode band is taken from a
       different moved volume according to the time the movement occurred.
    5. Inverse FFT; the magnitude image is the artifacted sample.

    Parameters
    ----------
    degrees: float or (float, float), default=10.0
        Range of the rotation angles (in degrees) of each movement,
        sampled per axis as :math:`\\theta \\sim \\mathcal{U}(a, b)`. If a
        single float :math:`n` is given, the range is :math:`(-n, n)`.
    translation: float or (float, float), default=10.0
        Range of the translations (in mm) of each movement, sampled per
        axis as :math:`t \\sim \\mathcal{U}(a, b)`. If a single float
        :math:`n` is given, the range is :math:`(-n, n)`.
    poisson_lambda: float, default=3.0
        Expectation :math:`\\lambda` of the Poisson distribution the
        number of movements is drawn from. At least one movement is
        always applied.
    voxel_size: float or (float, float, float), default=1.0
        Voxel size in mm, used to convert translations from mm to voxels.
    pe_axis: int or str, default=1
        Phase-encode axis. K-space is filled along this axis in time, so a
        movement time maps to a band of frequencies. Either an index in
        (0, 1, 2) or an anatomical label, "LR" (Left-Right),
        "AP" (Antero-Posterior) or "IS" (Inferior-Superior). In the latter
        case, a RAS-formatted affine matrix specifying the volume
        orientation must be provided when the transformation is called.
        Check `Nibabel documentation on image orientation
        <https://nipy.org/nibabel/coordinate_systems.html>`_.
    spline_order: int, default=3
        Order of the b-spline interpolation used to resample the moved
        volumes (in [0, 5]).
    kwargs: dict
        Keyword arguments given to :class:`nidl.transforms.Transform`
        (e.g. ``p``, the probability of applying the transform).

    References
    ----------
    .. [1] Shaw, R. et al. (2019). "MRI k-Space Motion Artefact
           Augmentation: Model Robustness and Task-Specific Uncertainty."
           Proceedings of Machine Learning Research, 102, 427-436.

    """

    def __init__(
        self,
        degrees: Union[float, tuple[float, float]] = 10.0,
        translation: Union[float, tuple[float, float]] = 10.0,
        poisson_lambda: float = 3.0,
        voxel_size: Union[float, tuple[float, float, float]] = 1.0,
        pe_axis: Union[int, str] = 1,
        spline_order: int = 3,
        **kwargs,
    ):
        super().__init__(**kwargs)

        if isinstance(degrees, numbers.Number):
            degrees = (-degrees, degrees)
        self.degrees = self._parse_range(degrees)
        if isinstance(translation, numbers.Number):
            translation = (-translation, translation)
        self.translation = self._parse_range(translation)
        self.poisson_lambda = self._parse_poisson_lambda(poisson_lambda)
        self.voxel_size = self._parse_voxel_size(voxel_size)
        self.pe_axis = self._parse_pe_axis(pe_axis)
        self.spline_order = self._parse_spline_order(spline_order)

    def apply_transform(
        self,
        data: TypeTransformInput,
        affine: Optional[np.ndarray] = None,
    ) -> TypeTransformInput:
        """Add a random motion artifact to the input volume.

        Parameters
        ----------
        data: np.ndarray or torch.Tensor
            Input volume with shape :math:`(C, H, W, D)` or
            :math:`(H, W, D)`. The same movement model is applied 
            independently to every channel.

        affine: np.ndarray of shape (4, 4) or None, default=None
            Affine transformation matrix of the input data in RAS format
            defining the orientation of the input image. This is typically
            given by Nibabel in this format. It is used only if ``pe_axis``
            is an anatomical label ("LR", "AP" or "IS") and ignored
            otherwise. If None, the identity matrix is used, assuming RAS
            orientation.

        Returns
        -------
        data: np.ndarray or torch.Tensor
            Corrupted volume. Output type and shape are the same as
            input.
        """
        data_is_tensor = isinstance(data, torch.Tensor)
        if data_is_tensor:
            dtype, device = data.dtype, data.device
            data = data.detach().cpu().numpy()

        pe_axis = self._resolve_pe_axis(affine)
        cumulative_affines, times = self._sample_movement_model()

        if data.ndim == 4:  # (C, H, W, D)
            corrupted_data = np.stack(
                [
                    self._add_motion(
                        channel, cumulative_affines, times, pe_axis
                    )
                    for channel in data
                ]
            )
        else:  # (H, W, D)
            corrupted_data = self._add_motion(
                data, cumulative_affines, times, pe_axis
            )

        corrupted_data = corrupted_data.astype(data.dtype)

        if data_is_tensor:
            corrupted_data = torch.as_tensor(
                corrupted_data, dtype=dtype, device=device
            )
        return corrupted_data

    def _sample_movement_model(
        self,
    ) -> tuple[list[np.ndarray], np.ndarray]:
        """Step 1: sample N rigid movements and their k-space times.

        Returns
        -------
        cumulative_affines: list of (N+1) np.ndarray of shape (4, 4)
            Cumulative poses, index 0 being the identity (no movement
            yet). The pose after movement :math:`k` is :math:`\\exp(
            \\sum_{i \\le k} \\log A_i)` (incremental log-Euclidean
            combination), so the motion accumulates/drifts rather than
            teleports.
        times: np.ndarray of shape (N,)
            Sorted uniform times in :math:`[0, 1]` of each movement,
            mapped later onto the phase-encode k-space axis.
        """
        # sample N movemenets from Poisson, ensuring at least 1 so an artifact
        # actually appears.
        n_movements = max(1, int(np.random.poisson(self.poisson_lambda)))

        cumulative_affines = [np.eye(4)]
        running_log = np.zeros((4, 4))
        for _ in range(n_movements):
            angles = np.deg2rad(np.random.uniform(*self.degrees, size=3))
            translation_mm = np.random.uniform(*self.translation, size=3)
            # Rigid 4x4 homogeneous transform (rotation + translation),
            # reusing the rotation builder of RandomAffine.
            affine = np.eye(4)
            affine[:3, :3] = RandomAffine._rotation_matrix(angles)
            affine[:3, 3] = translation_mm / self.voxel_size
            # Log-Euclidean: accumulate matrix logarithms so consecutive
            # poses drift instead of teleporting.
            running_log = running_log + logm(affine)
            cumulative_affines.append(np.real(expm(running_log)))

        # Times of the N movements, sorted (k-space filled monotonically).
        times = np.sort(np.random.uniform(0.0, 1.0, size=n_movements))

        return cumulative_affines, times

    def _add_motion(
        self,
        volume: np.ndarray,
        cumulative_affines: list[np.ndarray],
        times: np.ndarray,
        pe_axis: int,
    ) -> np.ndarray:
        """Steps 2-5: corrupt a single 3d volume with the given movements.

        Parameters
        ----------
        volume: np.ndarray of shape (H, W, D)
            Clean input volume.
        cumulative_affines: list of np.ndarray of shape (4, 4)
            Cumulative poses sampled by :meth:`_sample_movement_model`.
        times: np.ndarray
            Sorted movement times in :math:`[0, 1]`.
        pe_axis: int
            Phase-encode axis index in (0, 1, 2).

        Returns
        -------
        volume: np.ndarray of shape (H, W, D)
            Corrupted magnitude volume.
        """
        volume = volume.astype(np.float64)

        # K-space partition (one band per pose).
        band_masks = self._compute_kspace_band_masks(
            volume.shape, times, pe_axis
        )
        # Align number of poses to number of bands (robustness).
        n_poses = min(len(cumulative_affines), len(band_masks))
        cumulative_affines = cumulative_affines[:n_poses]
        band_masks = band_masks[:n_poses]

        # Step 2
        weights = self._signal_weights(volume, band_masks)
        demeaned = self._demean_affines(cumulative_affines, weights)

        # Steps 3 + 4: build the composite k-space band by band.
        composite_kspace = np.zeros(volume.shape, dtype=np.complex128)
        for affine, mask in zip(demeaned, band_masks):
            moved = self._apply_affine(volume, affine)  # Step 3
            kspace_moved = np.fft.fftshift(np.fft.fftn(moved))
            composite_kspace[mask] = kspace_moved[mask]  # Step 4

        # Step 5
        corrupted = np.fft.ifftn(np.fft.ifftshift(composite_kspace))
        return np.abs(corrupted)

    @staticmethod
    def _compute_kspace_band_masks(
        shape: tuple[int, ...], times: np.ndarray, pe_axis: int
    ) -> list[np.ndarray]:
        """Partition the phase-encode axis of k-space into bands.

        One contiguous band per pose of the volume: there are
        ``len(times) + 1`` poses (the identity pose plus N moved poses)
        and the band boundaries are placed at the movement times.

        Returns a list of boolean masks (each of full volume shape) that
        tile k-space along ``pe_axis`` with no overlap and full coverage.
        """
        n_poses = len(times) + 1
        n_pe = shape[pe_axis]

        # fftshift convention: the center of k-space (low frequencies) is
        # the middle index. Convert times in [0, 1] to integer split
        # points along the phase-encode axis.
        splits = [0] + [int(np.round(t * n_pe)) for t in times] + [n_pe]
        splits = sorted(set(splits))
        # Guarantee exactly n_poses bands by inserting midpoints in the
        # widest gaps.
        while len(splits) - 1 < n_poses:
            gaps = np.diff(splits)
            widest = int(np.argmax(gaps))
            midpoint = (splits[widest] + splits[widest + 1]) // 2
            if midpoint in splits:
                break
            splits.insert(widest + 1, midpoint)

        band_masks = []
        for low, high in zip(splits[:-1], splits[1:]):
            mask = np.zeros(shape, dtype=bool)
            slices = [slice(None)] * 3
            slices[pe_axis] = slice(low, high)
            mask[tuple(slices)] = True
            band_masks.append(mask)
        return band_masks

    @staticmethod
    def _signal_weights(
        volume: np.ndarray, band_masks: list[np.ndarray]
    ) -> np.ndarray:
        """Step 2 weights: image-domain signal carried by each band.

        Each weight is computed by masking the clean volume's k-space
        with the band, inverse-FFT and summing the magnitudes. Bands near
        the k-space center (low frequencies) carry most of the energy and
        thus get larger weights. Normalized to sum to 1.
        """
        kspace = np.fft.fftshift(np.fft.fftn(volume))
        weights = []
        for mask in band_masks:
            band_image = np.fft.ifftn(np.fft.ifftshift(mask * kspace))
            weights.append(np.sum(np.abs(band_image)))
        weights = np.array(weights, dtype=np.float64)
        return weights / weights.sum()

    @staticmethod
    def _demean_affines(
        cumulative_affines: list[np.ndarray], weights: np.ndarray
    ) -> list[np.ndarray]:
        """De-mean the poses w.r.t. their signal-weighted average.

        The average pose is computed in log-Euclidean space,
        :math:`A_{avg} = \\exp(\\sum_i w_i \\log A_i)`, then each pose is
        de-meaned as :math:`A_i' = A_{avg}^{-1} A_i`.
        """
        log_sum = np.zeros((4, 4))
        for weight, affine in zip(weights, cumulative_affines):
            log_sum += weight * logm(affine)
        average = np.real(expm(log_sum))
        average_inv = np.linalg.inv(average)
        return [average_inv @ affine for affine in cumulative_affines]

    def _apply_affine(
        self, volume: np.ndarray, affine: np.ndarray, pad: int = 10
    ) -> np.ndarray:
        """Step 3: resample the volume under a homogeneous affine.

        The transform is applied about the volume center with b-spline
        interpolation; the volume is edge-padded to limit edge artifacts
        and cropped back to its original shape.
        """
        padded = np.pad(volume, pad, mode="edge")
        center = (np.array(padded.shape) - 1) / 2.0

        rotation = affine[:3, :3]
        translation = affine[:3, 3]
        # Rotate about the center:
        # offset = center - rotation @ center (+ translation).
        offset = center - rotation @ center - translation

        moved = affine_transform(
            padded,
            rotation,
            offset=offset,
            order=self.spline_order,
            mode="nearest",
        )
        # Crop back to the original shape.
        slices = tuple(slice(pad, pad + size) for size in volume.shape)
        return moved[slices]

    def _resolve_pe_axis(self, affine: Optional[np.ndarray]) -> int:
        """Resolve the phase-encode axis to a voxel index in (0, 1, 2)."""
        if isinstance(self.pe_axis, int):
            return self.pe_axis
        affine = np.eye(4) if affine is None else affine
        return self.get_index_from_anat_label(self.pe_axis, affine)

    @staticmethod
    def get_index_from_anat_label(axis: str, affine: np.ndarray) -> int:
        """Returns the axis index corresponding to a given anatomical label.

        Parameters
        ----------
        axis : {'LR', 'AP', 'IS'}
            Anatomical axis label:

            - 'LR' for Left-Right (X axis)
            - 'AP' for Anterior-Posterior (Y axis)
            - 'IS' for Inferior-Superior (Z axis)

        affine : np.ndarray
            4x4 affine matrix defining the orientation of the volume.

        Returns
        -------
        int
            The index (0, 1, or 2) in voxel space corresponding to the
            requested anatomical axis.
        """
        anat_to_physical = {
            "LR": ("L", "R"),
            "AP": ("P", "A"),
            "IS": ("I", "S"),
        }

        if axis not in anat_to_physical:
            raise ValueError(
                f"Invalid axis '{axis}'. Must be one of 'LR', 'AP', 'IS'."
            )

        desired = anat_to_physical[axis]
        axcodes = aff2axcodes(affine)

        for i, code in enumerate(axcodes):
            if code in desired:
                return i

        raise ValueError(
            f"Could not find anatomical axis '{axis}' in affine matrix."
        )

    @staticmethod
    def _parse_poisson_lambda(poisson_lambda: float) -> float:
        if not isinstance(poisson_lambda, numbers.Number) or isinstance(
            poisson_lambda, bool
        ):
            raise TypeError(
                "`poisson_lambda` must be a number, got "
                f"{type(poisson_lambda)}"
            )
        if poisson_lambda <= 0:
            raise ValueError(
                f"`poisson_lambda` must be positive, got {poisson_lambda}"
            )
        return float(poisson_lambda)

    @staticmethod
    def _parse_voxel_size(
        voxel_size: Union[float, tuple[float, float, float]],
    ) -> np.ndarray:
        if isinstance(voxel_size, numbers.Number):
            voxel_size = 3 * (voxel_size,)
        voxel_size = np.asarray(voxel_size, dtype=np.float64)
        if voxel_size.shape != (3,):
            raise ValueError(
                "`voxel_size` must be a scalar or a length-3 sequence, "
                f"got shape {voxel_size.shape}"
            )
        if np.any(voxel_size <= 0):
            raise ValueError(
                f"`voxel_size` must be strictly positive, got {voxel_size}"
            )
        return voxel_size

    @staticmethod
    def _parse_pe_axis(pe_axis: Union[int, str]) -> Union[int, str]:
        if isinstance(pe_axis, bool):
            raise TypeError("`pe_axis` must be an int or str, not a bool")
        if isinstance(pe_axis, int):
            if pe_axis not in (0, 1, 2):
                raise ValueError(
                    f"`pe_axis` must be in {{0, 1, 2}}, got {pe_axis}"
                )
            return pe_axis
        if isinstance(pe_axis, str):
            if pe_axis not in ("LR", "AP", "IS"):
                raise ValueError(
                    "`pe_axis` must be in 'LR', 'AP' or 'IS', got "
                    f"'{pe_axis}'"
                )
            return pe_axis
        raise TypeError(
            f"`pe_axis` must be an int or str, got {type(pe_axis)}"
        )

    @staticmethod
    def _parse_spline_order(spline_order: int) -> int:
        if not isinstance(spline_order, int) or isinstance(
            spline_order, bool
        ):
            raise TypeError(
                f"`spline_order` must be an int, got {type(spline_order)}"
            )
        if not 0 <= spline_order <= 5:
            raise ValueError(
                f"`spline_order` must be in [0, 5], got {spline_order}"
            )
        return spline_order