from __future__ import annotations

import numbers
import random
import warnings
from typing import Union

import torch

from .....transforms import TypeTransformInput, VolumeTransform


class RandomBiasFieldFast(VolumeTransform):
    """Add a random MRI bias field artifact to a 3d volume
    (fast variant of RandomBiasField).

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
        if not data_is_tensor:
            data = torch.from_numpy(data)
        dtype, device = data.dtype, data.device

        if data.ndim == 4:  # (C, H, W, D)
            spatial_shape = data.shape[1:]
            if self.per_channel:
                biased_data = torch.stack(
                    [
                        self._apply_bias_field(
                            channel,
                            self._generate_bias_field(spatial_shape, device),
                            dtype,
                        )
                        for channel in data
                    ]
                )
            else:
                # Broadcast a single field across the channel dimension.
                biased_data = self._apply_bias_field(
                    data,
                    self._generate_bias_field(spatial_shape, device),
                    dtype,
                )
        else:  # (H, W, D)
            biased_data = self._apply_bias_field(
                data, self._generate_bias_field(data.shape, device), dtype
            )

        if not data_is_tensor:
            biased_data = biased_data.numpy()
        return biased_data

    @staticmethod
    def _apply_bias_field(
        data: torch.Tensor, log_field: torch.Tensor, dtype: torch.dtype
    ) -> torch.Tensor:
        """Multiplies `data` by ``exp(log_field)``.
        """
        field = log_field.exp_()
        if data.dtype == field.dtype and data.shape == field.shape:
            biased_data = field.mul_(data)
        else:
            biased_data = data * field
        return biased_data.to(dtype)

    def _generate_bias_field(
        self, shape: tuple[int, ...], device: torch.device
    ) -> torch.Tensor:
        """Returns the *log* bias field, as a float32 ``shape`` tensor."""
        order = self.order
        n_powers = order + 1

        powers = []
        for size in shape:
            coords = torch.arange(size, dtype=torch.float64) - size / 2 + 0.5
            coords_max = coords[-1].item()
            if coords_max > 0:
                coords /= coords_max
            # Coordinate powers, `basis[p, i] = coords[i] ** p`, transposed
            # Vandermonde matrix.
            basis = coords.unsqueeze(0) ** torch.arange(
                n_powers, dtype=torch.float64
            ).unsqueeze(1)
            powers.append(basis.to(device))
        x_powers, y_powers, z_powers = powers

        # Terms with `a + b + c > order` stay at zero and drop out of the sums.
        coeffs = torch.zeros(
            (n_powers, n_powers, n_powers), dtype=torch.float64
        )
        for a in range(n_powers):
            for b in range(n_powers - a):
                for c in range(n_powers - a - b):
                    coeffs[a, b, c] = random.uniform(*self.coefficients)
        coeffs = coeffs.to(device)

        # log_field[i, j, k] = sum_{a, b, c} coeffs[a, b, c]
        #                     * x[i] ** a * y[j] ** b * z[k] ** c
        # contracted one axis at a time.
        field = torch.tensordot(coeffs, x_powers, dims=([0], [0]))
        # (order + 1, H, W)
        field = torch.tensordot(field, y_powers, dims=([0], [0]))
        # (H, W, D)
        field = torch.tensordot(
            field.to(torch.float32),
            z_powers.to(torch.float32),
            dims=([0], [0]),
        )
        return field

    @staticmethod
    def _parse_order(order: int) -> int:
        if not isinstance(order, int) or isinstance(order, bool):
            raise TypeError(f"`order` must be an int, got {type(order)}")
        if order < 0:
            raise ValueError(f"`order` must be non-negative, got {order}")
        return order
