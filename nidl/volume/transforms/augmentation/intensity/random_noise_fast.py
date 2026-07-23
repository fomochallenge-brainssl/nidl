##########################################################################
# NSAp - Copyright (C) CEA, 2025
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################
from __future__ import annotations

import numbers
import random
from typing import Union

import numpy as np
import torch

from .....transforms import TypeTransformInput, VolumeTransform


class RandomNoiseFast(VolumeTransform):
    """Add Gaussian noise to input data with random parameters.

    The noise is sampled from a Gaussian distribution with randomly selected
    mean and standard deviation. With ``noise_downsampling_factor=1``, one
    independent noise value is generated per voxel. For values greater than 1,
    noise is generated on a lower-resolution spatial grid and each value is
    repeated across a spatial block, resulting in spatially correlated noise.

    The input data can have any shape with the last three dimensions
    representing the spatial dimensions, and must be a :class:`numpy.ndarray`
    or :class:`torch.Tensor`. The output preserves the input type and shape.

    Parameters
    ----------
    mean: float or (float, float), default=0.0
        Mean :math:`\\mu` of the Gaussian distribution. If two values
        :math:`(a, b)` are given, then :math:`\\mu \\sim \\mathcal{U}(a, b)`.

    std: (float, float), default=(0.1, 1.0)
        Range of the standard deviation :math:`(a, b)` of the Gaussian
        distribution. The standard deviation is sampled as
        :math:`\\sigma \\sim \\mathcal{U}(a, b)`.

    noise_downsampling_factor: int, default=2
        Factor used to reduce the spatial resolution of the generated noise.
        Must be at least 1. A value of 1 generates one noise value per voxel.
        For values greater than 1, noise values are repeated across spatial
        blocks. Spatial dimensions do not need to be divisible by the factor.

    kwargs: dict
        Keyword arguments.
    """

    def __init__(
        self,
        mean: Union[float, tuple[float, float]] = 0.0,
        std: tuple[float, float] = (0.1, 1.0),
        noise_downsampling_factor: int = 2,
        **kwargs,
    ):
        super().__init__(**kwargs)

        if isinstance(mean, numbers.Number):
            mean = (mean, mean)

        self.mean = self._parse_range(mean)
        self.std = self._parse_range(std, check_min=0)

        if (
            not isinstance(noise_downsampling_factor, int)
            or isinstance(noise_downsampling_factor, bool)
            or noise_downsampling_factor < 1
        ):
            raise ValueError(
                "noise_downsampling_factor must be an integer >= 1."
            )

        self.noise_downsampling_factor = noise_downsampling_factor

    def apply_transform(
        self,
        data: TypeTransformInput,
    ) -> TypeTransformInput:
        """Add Gaussian noise to the input data.

        The noise is sampled on a potentially downsampled spatial grid
        and repeated across spatial blocks. See the class docstring for
        details.
        """
        factor = self.noise_downsampling_factor

        mean = random.uniform(*self.mean)
        std = random.uniform(*self.std)

        spatial_shape = data.shape[-3:]

        noise_spatial_shape = tuple(
            (size + factor - 1) // factor
            for size in spatial_shape
        )

        noise_shape = (
            *data.shape[:-3],
            *noise_spatial_shape,
        )

        if isinstance(data, torch.Tensor):
            noise = torch.empty(
                noise_shape,
                dtype=data.dtype,
                device=data.device,
            )
            noise.normal_(mean=mean, std=std)

            if factor > 1:
                noise = noise.repeat_interleave(factor, dim=-3)
                noise = noise.repeat_interleave(factor, dim=-2)
                noise = noise.repeat_interleave(factor, dim=-1)

                noise = noise[
                    ...,
                    :spatial_shape[0],
                    :spatial_shape[1],
                    :spatial_shape[2],
                ]

            return data + noise

        noise = np.random.normal(
            mean,
            std,
            size=noise_shape,
        ).astype(data.dtype)

        if factor > 1:
            noise = np.repeat(noise, factor, axis=-3)
            noise = np.repeat(noise, factor, axis=-2)
            noise = np.repeat(noise, factor, axis=-1)

            noise = noise[
                ...,
                :spatial_shape[0],
                :spatial_shape[1],
                :spatial_shape[2],
            ]

        return data + noise
