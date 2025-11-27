import mlx.core as mx
from mlx import nn


class ConvIn(nn.Module):
    def __init__(self, in_channels: int = 3, out_channels: int = 128):
        super().__init__()
        self.conv2d = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
        )

    def __call__(self, input_array: mx.array) -> mx.array:
        input_array = mx.transpose(input_array, (0, 2, 3, 1))
        output = self.conv2d(input_array)
        return mx.transpose(output, (0, 3, 1, 2))
