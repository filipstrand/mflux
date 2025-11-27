import mlx.core as mx
from mlx import nn
from mlx.core.fast import scaled_dot_product_attention

from mflux.models.common.config import ModelConfig


class Attention(nn.Module):
    def __init__(self):
        super().__init__()
        self.group_norm = nn.GroupNorm(32, 512, pytorch_compatible=True)
        self.to_q = nn.Linear(512, 512)
        self.to_k = nn.Linear(512, 512)
        self.to_v = nn.Linear(512, 512)
        self.to_out = [nn.Linear(512, 512)]

    def __call__(self, input_array: mx.array) -> mx.array:
        input_array = mx.transpose(input_array, (0, 2, 3, 1))

        B, H, W, C = input_array.shape

        y = self.group_norm(input_array.astype(mx.float32)).astype(ModelConfig.precision)

        queries = self.to_q(y).reshape(B, H * W, 1, C)
        keys = self.to_k(y).reshape(B, H * W, 1, C)
        values = self.to_v(y).reshape(B, H * W, 1, C)

        queries = mx.transpose(queries, (0, 2, 1, 3))
        keys = mx.transpose(keys, (0, 2, 1, 3))
        values = mx.transpose(values, (0, 2, 1, 3))

        scale = 1 / mx.sqrt(queries.shape[-1])
        y = scaled_dot_product_attention(queries, keys, values, scale=scale)

        y = mx.transpose(y, (0, 2, 1, 3)).reshape(B, H, W, C)

        y = self.to_out[0](y)
        output_tensor = input_array + y

        return mx.transpose(output_tensor, (0, 3, 1, 2))
