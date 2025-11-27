import mlx.core as mx
from mlx import nn

from mflux.models.common.config import ModelConfig


class ConvNormOut(nn.Module):
    def __init__(self, channels: int = 128):
        super().__init__()
        self.norm = nn.GroupNorm(
            num_groups=32,
            dims=channels,
            eps=1e-6,
            affine=True,
            pytorch_compatible=True,
        )

    def __call__(self, input_array: mx.array) -> mx.array:
        input_array = mx.transpose(input_array, (0, 2, 3, 1))
        hidden_states = self.norm(input_array.astype(mx.float32)).astype(ModelConfig.precision)
        return mx.transpose(hidden_states, (0, 3, 1, 2))
