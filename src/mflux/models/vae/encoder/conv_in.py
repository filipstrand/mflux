import mlx.core as mx
import mlx.nn as nn


class ConvIn(nn.Module):

    def __init__(self):
        super().__init__()
        self.conv2d = nn.Conv2d(
            in_channels=3,
            out_channels=128,
            kernel_size=3,
            stride=1,
            padding=1,
        )

    def forward(self, input_array: mx.array) -> mx.array:
        input_array = mx.transpose(input_array, (0, 2, 3, 1))
        return mx.transpose(self.conv2d(input_array), (0, 3, 1, 2))
