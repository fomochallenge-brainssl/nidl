from typing import Union
import math
import torch

from .....transforms import VolumeTransform
from .resize import Resize
from .crop_or_pad import CropOrPad

# Choice of interpolation and padding mode might be critical for segmentation mask
class Letterbox(VolumeTransform):
    def __init__(
        self,
        target_shape: Union[int, tuple[int, int, int]],
        padding_mode: str = "constant",
        constant_values: Union[float, tuple[float, float, float]] = 0.0,
        interpolation: str = "linear"
    ):
        super().__init__()

        self.target_shape = self._parse_shape(target_shape, length=3)
        if self.target_shape[0] == self.target_shape[1] and self.target_shape[1] == self.target_shape[2]:
            self.interpolation = interpolation
            self.padding_transform = CropOrPad(self.target_shape, padding_mode, constant_values)
        else:
            raise ValueError(f"Target shape must be isotropic\n")
    
    def compute_intermediate_shape(self, original_shape):
        longest_dim = max(original_shape)
        scale_factor = self.target_shape[0] / longest_dim
        resized_shape = tuple(int(round(dim * scale_factor)) for dim in original_shape)

        return resized_shape
    
    def apply_transform(self, data_parsed, *args, **kwargs):
        original_shape = data_parsed.shape

        if len(original_shape) == 4:
            in_shape = original_shape[1:]
            channels_count = original_shape[0]
        else:
            in_shape = original_shape
            channels_count = 1
            data_parsed = data_parsed.unsqueeze(0)

        # Apply the transform across all channels
        output_image = []
        for i in range(channels_count):
            # 1. Compute scale factor (same for all dimensions to preserve ratio)
            resized_shape = self.compute_intermediate_shape(in_shape)

            # 2. Resize image
            resize_transform = Resize(resized_shape, self.interpolation)
            resized_image = resize_transform(data_parsed[i])

            # 3. Padding to target shape
            padded_image = self.padding_transform(resized_image)

            output_image.append(padded_image)
        
        output_image = torch.stack(output_image, dim=0)

        # Remove channel dimension
        if len(original_shape) == 3:
            output_image = output_image.squeeze(0)


        return output_image
