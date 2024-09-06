import mlx.core as mx
import mlx.nn as nn

from mflux.config.config import Config


class ConvNormOut(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm = nn.GroupNorm(
            num_groups=32,
            dims=128,
            eps=1e-6,
            affine=True,
            pytorch_compatible=True
        )

    def forward(self, input_array: mx.array) -> mx.array:
        input_array = mx.transpose(input_array, (0, 2, 3, 1))
        hidden_states = self.norm(input_array.astype(mx.float32)).astype(Config.precision)
        return mx.transpose(hidden_states, (0, 3, 1, 2))
