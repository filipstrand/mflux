import mlx.core as mx
from mlx import nn

from mflux.models.qwen.model.qwen_vae.qwen_image_causal_conv_3d import QwenImageCausalConv3D


class QwenImageResample3D(nn.Module):
    def __init__(self, dim: int, mode: str):
        super().__init__()
        self.dim = dim
        self.mode = mode
        if mode == "upsample3d":
            self.time_conv = QwenImageCausalConv3D(dim, dim * 2, kernel_size=(3, 1, 1), stride=1, padding=(1, 0, 0))
            self.resample_conv = nn.Conv2d(dim, dim // 2, kernel_size=3, stride=1, padding=1)
        elif mode == "upsample2d":
            self.resample_conv = nn.Conv2d(dim, dim // 2, kernel_size=3, stride=1, padding=1)
        elif mode == "downsample3d":
            self.time_conv = QwenImageCausalConv3D(dim, dim, kernel_size=(3, 1, 1), stride=(2, 1, 1), padding=(0, 0, 0))
            self.resample_conv = nn.Conv2d(dim, dim, kernel_size=3, stride=2, padding=0)
        elif mode == "downsample2d":
            self.resample_conv = nn.Conv2d(dim, dim, kernel_size=3, stride=2, padding=0)
        else:
            raise ValueError(f"Unsupported resample mode: {mode}")

    def __call__(self, x: mx.array) -> mx.array:
        b, c, t, h, w = x.shape
        t = x.shape[2]
        x = mx.transpose(x, (0, 2, 1, 3, 4))
        x = mx.reshape(x, (b * t, c, h, w))
        x = mx.transpose(x, (0, 2, 3, 1))
        if self.mode in ["upsample3d", "upsample2d"]:
            x = self._up_sample_nearest_2x(x)

        if self.mode in ["downsample2d", "downsample3d"]:
            x = mx.pad(x, [(0, 0), (0, 1), (0, 1), (0, 0)])

        x = self.resample_conv(x)
        x = mx.transpose(x, (0, 3, 1, 2))
        new_c = x.shape[1]
        new_h, new_w = x.shape[2], x.shape[3]
        x = mx.reshape(x, (b, t, new_c, new_h, new_w))
        x = mx.transpose(x, (0, 2, 1, 3, 4))
        return x

    @staticmethod
    def _up_sample_nearest_2x(x: mx.array) -> mx.array:
        B_T, H, W, C = x.shape
        x = mx.repeat(x, 2, axis=1)
        x = mx.repeat(x, 2, axis=2)
        return x
