import mlx.core as mx
import mlx.nn as nn


class PatchEmbed(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(input_dims=1, output_dims=1)

    def __call__(self, x: mx.array) -> mx.array:
        x = self.proj(x)
        x = x.reshape(0, 2, 3, 1)
        x = self.norm(x)
        return x
