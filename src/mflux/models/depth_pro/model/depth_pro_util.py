import math

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from PIL import Image


class DepthProUtil:
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

    @staticmethod
    def interpolate(x: mx.array, size=None, scale_factor=None):
        x_np = np.array(x)
        original_ndim = x_np.ndim

        if original_ndim == 3:
            C, H_in, W_in = x_np.shape
            x_proc = np.expand_dims(x_np, 0)
        elif original_ndim == 4:
            _, C, H_in, W_in = x_np.shape
            x_proc = x_np
        else:
            raise ValueError(f"Unsupported input shape: {x_np.shape}. Must be 3D (C,H,W) or 4D (B,C,H,W).")

        if size is not None:
            H_out, W_out = size
        elif scale_factor is not None:
            H_out, W_out = int(H_in * scale_factor), int(W_in * scale_factor)
        else:
            return x

        B_proc, C_proc, _, _ = x_proc.shape

        result_proc = np.zeros((B_proc, C_proc, H_out, W_out), dtype=x_np.dtype)

        for b in range(B_proc):
            for c_idx in range(C_proc):
                channel_img_np = x_proc[b, c_idx]
                pil_img = Image.fromarray(channel_img_np)
                resized_pil_img = pil_img.resize((W_out, H_out), Image.NEAREST)
                result_proc[b, c_idx] = np.array(resized_pil_img)

        if original_ndim == 3:
            final_result_np = result_proc.squeeze(0)
        else:
            final_result_np = result_proc

        return mx.array(final_result_np)

    @staticmethod
    def apply_conv(x: mx.array, conv_module: nn.Module) -> mx.array:
        x = mx.transpose(x, (0, 2, 3, 1))
        x = conv_module(x)
        return mx.transpose(x, (0, 3, 1, 2))
