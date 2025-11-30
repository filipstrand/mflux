import mlx.core as mx
from mlx import nn


class UpSampler(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
        )

    def __call__(self, input_array: mx.array) -> mx.array:
        input_array = mx.transpose(input_array, (0, 2, 3, 1))
        hidden_states = self._upsample_nearest(input_array, scale=2)
        hidden_states = self.conv(hidden_states)
        return mx.transpose(hidden_states, (0, 3, 1, 2))

    @staticmethod
    def _upsample_nearest(x: mx.array, scale: int = 2) -> mx.array:
        B, H, W, C = x.shape
        x = mx.broadcast_to(x[:, :, None, :, None, :], (B, H, scale, W, scale, C))
        x = x.reshape(B, H * scale, W * scale, C)
        return x
