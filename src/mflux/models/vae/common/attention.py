import mlx.core as mx
from mlx import nn

from mflux.config.config import Config


class Attention(nn.Module):

    def __init__(self):
        super().__init__()
        self.group_norm = nn.GroupNorm(32, 512, pytorch_compatible=True)
        self.to_q = nn.Linear(512, 512)
        self.to_k = nn.Linear(512, 512)
        self.to_v = nn.Linear(512, 512)
        self.to_out = [nn.Linear(512, 512)]

    def forward(self, input_array: mx.array) -> mx.array:
        input_array = mx.transpose(input_array, (0, 2, 3, 1))

        B, H, W, C = input_array.shape

        y = self.group_norm(input_array.astype(mx.float32)).astype(Config.precision)

        queries = self.to_q(y).reshape(B, H * W, C)
        keys = self.to_k(y).reshape(B, H * W, C)
        values = self.to_v(y).reshape(B, H * W, C)

        scale = 1 / mx.sqrt(queries.shape[-1])
        scores = (queries * scale) @ keys.transpose(0, 2, 1)
        attn = mx.softmax(scores, axis=-1)
        y = (attn @ values).reshape(B, H, W, C)

        y = self.to_out[0](y)
        output_tensor = input_array + y

        return mx.transpose(output_tensor, (0, 3, 1, 2))
