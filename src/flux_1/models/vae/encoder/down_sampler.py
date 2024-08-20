import mlx.core as mx

from mlx import nn


class DownSampler(nn.Module):

    def __init__(self, conv_in: int, conv_out: int):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels=conv_in,
            out_channels=conv_out,
            kernel_size=(3, 3),
            stride=(2, 2),
        )

    def forward(self, input_array: mx.array) -> mx.array:
        hidden_states = mx.pad(input_array, ((0, 0), (0, 0), (0, 1), (0, 1)))
        hidden_states = mx.transpose(hidden_states, (0, 2, 3, 1))
        hidden_state = self.conv(hidden_states)
        return mx.transpose(hidden_state, (0, 3, 1, 2))
