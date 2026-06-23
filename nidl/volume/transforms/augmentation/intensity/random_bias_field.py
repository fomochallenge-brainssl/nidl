from __future__ import annotations

import numbers
import random
from typing import Union

import numpy as np
import torch

from .....transforms import TypeTransformInput, VolumeTransform


class RandomBiasField(VolumeTransform):
    """Add a random MRI bias field artifact to a 3d volume.

    Following Van Leemput et al. (1999) [1]_, the bias field is modeled as the
    exponential of a linear combination of polynomial basis functions, whose
    coefficients are randomly sampled:

    .. math::

        b(x, y, z) = \\exp\\left( \\sum_{\\substack{i, j, k \\geq 0 \\\\
        i + j + k \\leq n}} c_{ijk}\\, x^i y^j z^k \\right)

    where :math:`n` is the polynomial order, :math:`(x, y, z)` are voxel
    coordinates normalized to :math:`[-1, 1]` and :math:`c_{ijk} \\sim
    \\mathcal{U}(a, b)` are the random coefficients. The volume is then
    multiplied by :math:`b`.

    Parameters
    ----------
    coefficients: float or (float, float), default=0.5
        Range of the polynomial coefficients :math:`c_{ijk} \\sim
        \\mathcal{U}(a, b)`. If a single float :math:`n` is given, the range
        is :math:`(-n, n)`. Larger magnitudes yield stronger inhomogeneity.
    order: int, default=3
        Order :math:`n` of the polynomial basis functions. Must be a
        non-negative integer.
    per_channel: bool, default=True
        If ``True``, an independent bias field is sampled for each channel.
    kwargs: dict
        Keyword arguments given to :class:`nidl.transforms.Transform`
        (e.g. ``p``, the probability of applying the transform).

    References
    ----------
    .. [1] Van Leemput, K. et al. (1999). "Automated model-based bias field
           correction of MR images of the brain." IEEE Transactions on
           Medical Imaging, 18(10), 885-896.
        

    Examples
    --------
    >>> import torch
    >>> from nidl.volume.transforms.augmentation.intensity import (
    ...     RandomBiasField)
    >>> volume = torch.rand(1, 64, 64, 64)  # shape (C, H, W, D)
    >>> transform = RandomBiasField(coefficients=0.5, order=3)
    >>> biased = transform(volume)  # shape (1, 64, 64, 64)
    """

    def __init__(
        self,
        coefficients: Union[float, tuple[float, float]] = 0.5,
        order: int = 3,
        per_channel: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)

        if isinstance(coefficients, numbers.Number):
            coefficients = (-coefficients, coefficients)
        self.coefficients = self._parse_range(coefficients)
        self.order = self._parse_order(order)
        self.per_channel = per_channel

    def apply_transform(self, data: TypeTransformInput) -> TypeTransformInput:
        data_is_tensor = isinstance(data, torch.Tensor)
        if data_is_tensor:
            dtype, device = data.dtype, data.device
            data = data.detach().cpu().numpy()

        if data.ndim == 4:  # (C, H, W, D)
            spatial_shape = data.shape[1:]
            if self.per_channel:
                biased_data = np.stack(
                    [
                        channel * self._generate_bias_field(spatial_shape)
                        for channel in data
                    ]
                )
            else:
                # Broadcast a single field across the channel dimension.
                biased_data = data * self._generate_bias_field(spatial_shape)
        else:  # (H, W, D)
            biased_data = data * self._generate_bias_field(data.shape)

        biased_data = biased_data.astype(data.dtype)

        if data_is_tensor:
            biased_data = torch.as_tensor(
                biased_data, dtype=dtype, device=device
            )
        return biased_data

    def _generate_bias_field(self, shape: tuple[int, ...]) -> np.ndarray:
        shape = np.asarray(shape)
        half_shape = shape / 2

        # One coordinate vector per axis, centered on the volume.
        ranges = [np.arange(-n, n) + 0.5 for n in half_shape]
        # `indexing="ij"` keeps the axes in (H, W, D) order (no transpose).
        meshes = np.meshgrid(*ranges, indexing="ij")
        # Normalize each axis to [-1, 1] so coefficients are scale-invariant.
        for mesh in meshes:
            mesh_max = mesh.max()
            if mesh_max > 0:
                mesh /= mesh_max
        x_mesh, y_mesh, z_mesh = meshes

        log_field = np.zeros(shape, dtype=np.float64)
        for a in range(self.order + 1):
            for b in range(self.order + 1 - a):
                for c in range(self.order + 1 - a - b):
                    coefficient = random.uniform(*self.coefficients)
                    log_field += (
                        coefficient
                        * x_mesh**a
                        * y_mesh**b
                        * z_mesh**c
                    )
        bias_field = np.exp(log_field).astype(np.float32)
        return bias_field

    @staticmethod
    def _parse_order(order: int) -> int:
        if not isinstance(order, int) or isinstance(order, bool):
            raise TypeError(f"`order` must be an int, got {type(order)}")
        if order < 0:
            raise ValueError(f"`order` must be non-negative, got {order}")
        return order
