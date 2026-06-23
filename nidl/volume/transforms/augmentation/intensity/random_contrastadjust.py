from __future__ import annotations

import numbers
import random
from typing import Optional, Union

import numpy as np
import torch

from .....transforms import TypeTransformInput, VolumeTransform


class RandomContrastAdjust(VolumeTransform):
    """Randomly adjust the brightness and contrast of a 3d volume,
    using the linear transformation:
    .. math::

        p_\\text{out} = \\alpha \\cdot p_\\text{in} + \\beta

    where :math:`p_\\text{in}` (resp. :math:`p_\\text{out}`) is the input
    (resp. output) voxel intensity, :math:`\\alpha` is the contrast factor
    (gain) and :math:`\\beta` is the brightness factor (bias).

    Parameters
    ----------
    contrast_factor: (float, float), default=(0.75, 1.25)
        Contrast factor (gain) :math:`\\alpha`, sampled
        :math:`\\alpha \\sim \\mathcal{U}(a, b)`. A factor of 1.0 leaves the
        contrast unchanged, a factor in :math:`[0, 1)` reduces it and a factor
        greater than 1.0 increases it. Both bounds must be :math:`\\ge 0`
        (a negative gain would invert intensities, not adjust contrast).
    brightness_factor: float or (float, float), default=0.0
        Brightness factor (bias) :math:`\\beta`. If two values
        :math:`(a, b)` are given, then
        :math:`\\beta \\sim \\mathcal{U}(a, b)`. If a single number ``b`` is
        given, the symmetric range ``(-b, b)`` is used. A factor of 0.0 leaves
        the brightness unchanged.
    output_range: (float, float) or None, default=None
        If a tuple :math:`(low, high)` is given, the output is clipped to
        this range to discard impossible intensity values introduced by the
        adjustment (e.g. ``(0, 1)`` for min-max rescaled inputs). If
        ``None`` (default), no clipping is performed.
    kwargs: dict
        Keyword arguments.

    Notes
    -----
    This transformation can be used to simulate different lighting
    conditions and scanner variability. Assumes intensity-normalized
    (zero-centered) inputs, so the contrast is scaled around 0 rather than
    around a per-volume statistic.

    Examples
    --------
    >>> import torch
    >>> from nidl.volume.transforms.augmentation.intensity import (
    ...     RandomContrastAdjust)
    >>> volume = torch.randn(1, 64, 64, 64)  # z-scored, shape (C, H, W, D)
    >>> transform = RandomContrastAdjust(
    ...     contrast_factor=(0.8, 1.2), brightness_factor=(-0.1, 0.1))
    >>> adjusted = transform(volume)  # shape (1, 64, 64, 64)
    """

    def __init__(
        self,
        contrast_factor: tuple[float, float] = (0.75, 1.25),
        brightness_factor: Union[float, tuple[float, float]] = 0.0,
        output_range: Optional[tuple[float, float]] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.contrast_factor = self._parse_range(contrast_factor, check_min=0)

        if isinstance(brightness_factor, numbers.Number):
            if brightness_factor < 0:
                raise ValueError(
                    "A single brightness_factor must be non-negative, got "
                    f"{brightness_factor}"
                )
            brightness_factor = (-brightness_factor, brightness_factor)
        self.brightness_factor = self._parse_range(brightness_factor)

        if output_range is not None:
            output_range = self._parse_range(output_range)
        self.output_range = output_range

    def apply_transform(self, data: TypeTransformInput) -> TypeTransformInput:
        """Adjust the brightness and contrast of the input.

        Parameters
        ----------
        data: np.ndarray or torch.Tensor
            The input volume.

        Returns
        -------
        data: np.ndarray or torch.Tensor
            Brightness/contrast adjusted volume. Output type and shape are
            the same as input.
        """
        alpha = random.uniform(*self.contrast_factor)
        beta = random.uniform(*self.brightness_factor)

        data_is_tensor = isinstance(data, torch.Tensor)
        if data_is_tensor:
            dtype, device = data.dtype, data.device
            data = data.detach().cpu().numpy()

        adjusted_data = alpha * data + beta

        if self.output_range is not None:
            adjusted_data = np.clip(adjusted_data, *self.output_range)

        adjusted_data = adjusted_data.astype(data.dtype)

        if data_is_tensor:
            adjusted_data = torch.as_tensor(
                adjusted_data, dtype=dtype, device=device
            )

        return adjusted_data
