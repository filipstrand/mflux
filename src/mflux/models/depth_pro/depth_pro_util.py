import math

import mlx.core as mx
import numpy as np
import torch
import torch.nn.functional as F


class DepthProUtil:
    @staticmethod
    def create_pyramid(x: mx.array) -> (mx.array, mx.array, mx.array):
        x0 = x
        x_np = np.array(x)
        x_torch = torch.from_numpy(x_np)
        x1_torch = F.interpolate(x_torch, size=None, scale_factor=0.5, mode="bilinear", align_corners=False)
        x2_torch = F.interpolate(x_torch, size=None, scale_factor=0.25, mode="bilinear", align_corners=False)
        x1 = mx.array(x1_torch.numpy())
        x2 = mx.array(x2_torch.numpy())
        return x0, x1, x2

    @staticmethod
    def split(x: mx.array, overlap_ratio: float = 0.25) -> mx.array:
        patch_size = 384
        patch_stride = int(patch_size * (1 - overlap_ratio))

        image_size = x.shape[-1]
        steps = int(math.ceil((image_size - patch_size) / patch_stride)) + 1

        x_patch_list = []
        for j in range(steps):
            j0 = j * patch_stride
            j1 = j0 + patch_size

            for i in range(steps):
                i0 = i * patch_stride
                i1 = i0 + patch_size
                x_patch_list.append(x[..., j0:j1, i0:i1])

        return mx.concatenate(x_patch_list, axis=0)
