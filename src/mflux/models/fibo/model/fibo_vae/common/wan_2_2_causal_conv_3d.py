import mlx.core as mx
from mlx import nn


class Wan2_2_CausalConv3d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        name: str | None = None,
    ):
        super().__init__()
        self.conv3d = nn.Conv3d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=0,
        )
        self.padding = padding
        self.stride = stride
        self.kernel_size = kernel_size
        self.name = name or f"conv3d_{in_channels}to{out_channels}"

    def __call__(self, x: mx.array) -> mx.array:
        pad_t = pad_h = pad_w = self.padding
        if pad_t > 0 or pad_h > 0 or pad_w > 0:
            pad_spec = [
                (0, 0),
                (0, 0),
                (2 * pad_t, 0),
                (pad_h, pad_h),
                (pad_w, pad_w),
            ]
            x = mx.pad(x, pad_spec)
        x = mx.transpose(x, (0, 2, 3, 4, 1))
        x = self.conv3d(x)
        x = mx.transpose(x, (0, 4, 1, 2, 3))
        return x
