import mlx.core as mx

from mlx import nn


class UpSampler(nn.Module):

    def __init__(self, conv_in: int, conv_out: int):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels=conv_in,
            out_channels=conv_out,
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=(1, 1),
        )

    def forward(self, input_array: mx.array) -> mx.array:
        input_array = mx.transpose(input_array, (0, 2, 3, 1))
        hidden_states = UpSampler.up_sample_nearest(input_array)
        hidden_state = self.conv(hidden_states)
        return mx.transpose(hidden_state, (0, 3, 1, 2))

    @staticmethod
    def up_sample_nearest(x: mx.array, scale: int = 2):
        B, H, W, C = x.shape
        x = mx.broadcast_to(x[:, :, None, :, None, :], (B, H, scale, W, scale, C))
        x = x.reshape(B, H * scale, W * scale, C)
        return x
