from typing import Union
import math
import torch

from .....transforms import VolumeTransform
from .resize import Resize
from .crop_or_pad import CropOrPad


class InverseLetterbox(VolumeTransform):
    def __init__(
        self, 
        standard_shape: Union[int, tuple[int, int, int]],
        interpolation: str = "linear"
    ):
        super().__init__()

        self.standard_shape = self._parse_shape(standard_shape, length=3)
        if self.standard_shape[0] == self.standard_shape[1] and self.standard_shape[1] == self.standard_shape[2]:
            self.interpolation = interpolation
        else:
            raise ValueError("standard_shape must be isotropic\n")
    
    def compute_intermediate_shape(self, original_shape):
        longest_dim = max(original_shape)
        scale_factor = self.standard_shape[0] / longest_dim
        resized_shape = tuple(int(round(dim * scale_factor)) for dim in original_shape)

        return resized_shape
    
    def apply_transform(self, data_parsed, original_shape, *args, **kwargs):  
        in_shape = data_parsed.shape
              
        if len(in_shape) == 4:
            channels_count = data_parsed.shape[0]
        else:
            channels_count = 1
            data_parsed = data_parsed.unsqueeze(0)

        # Apply the same transform to all channels
        output_image = []
        for i in range(channels_count):
            # 1. Compute post-scaling shape
            resized_shape = self.compute_intermediate_shape(original_shape)

            # 2. Crop back from standard shape to rescaled shape
            cropping_transform = CropOrPad(resized_shape)
            cropped_image = cropping_transform(data_parsed[i])

            # 3. Re-size to original shape
            resize_transform = Resize(original_shape, self.interpolation)
            original_image = resize_transform(cropped_image)

            output_image.append(original_image)
        
        output_image = torch.stack(output_image, dim = 0)

        if len(in_shape) == 3:
            output_image = output_image.squeeze(0)

        return output_image
