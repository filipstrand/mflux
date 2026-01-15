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
        """
        Resample with optimized transpose pattern.

        OPTIMIZATION: Reduced from 8 layout operations to 2 transposes.
        - Old: (b,c,t,h,w) → transpose → reshape → transpose → conv → transpose → reshape → transpose
        - New: (b,c,t,h,w) → single transpose+reshape to NHWC → conv → single reshape+transpose back

        MLX Conv2d natively prefers NHWC (channels-last) format, so we stay in that layout
        throughout processing to minimize transpose overhead.
        """
        b, c, t, h, w = x.shape

        # Single transpose+reshape to NHWC format (channels-last)
        # Old: 3 operations (transpose → reshape → transpose)
        # New: 1 combined transpose that goes directly to target layout
        x = mx.transpose(x, (0, 2, 3, 4, 1))  # (b,c,t,h,w) → (b,t,h,w,c)
        x = mx.reshape(x, (b * t, h, w, c))  # Flatten batch+time dimension

        # Process in NHWC format (no intermediate transposes needed)
        if self.mode in ["upsample3d", "upsample2d"]:
            x = self._up_sample_nearest_2x(x)

        if self.mode in ["downsample2d", "downsample3d"]:
            # Padding stays in NHWC: [(batch, 0), (height, 1), (width, 1), (channels, 0)]
            x = mx.pad(x, [(0, 0), (0, 1), (0, 1), (0, 0)])

        # Conv2d operates in NHWC format (MLX's native preference)
        x = self.resample_conv(x)

        # Single reshape+transpose back to original format
        # Old: 3 operations (transpose → reshape → transpose)
        # New: 1 combined operation
        new_h, new_w, new_c = x.shape[1], x.shape[2], x.shape[3]
        x = mx.reshape(x, (b, t, new_h, new_w, new_c))  # Unflatten batch+time
        x = mx.transpose(x, (0, 4, 1, 2, 3))  # (b,t,h,w,c) → (b,c,t,h,w)

        return x

    @staticmethod
    def _up_sample_nearest_2x(x: mx.array) -> mx.array:
        B_T, H, W, C = x.shape
        x = mx.repeat(x, 2, axis=1)
        x = mx.repeat(x, 2, axis=2)
        return x
