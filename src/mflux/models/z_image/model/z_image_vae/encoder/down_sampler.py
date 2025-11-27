import mlx.core as mx
from mlx import nn


class DownSampler(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=2,
            padding=0,
        )

    def __call__(self, input_array: mx.array) -> mx.array:
        hidden_states = mx.pad(input_array, ((0, 0), (0, 0), (0, 1), (0, 1)))
        hidden_states = mx.transpose(hidden_states, (0, 2, 3, 1))
        hidden_states = self.conv(hidden_states)
        return mx.transpose(hidden_states, (0, 3, 1, 2))
