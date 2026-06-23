"""SwiGLU feed-forward for Krea-2.

``mlpdim = round_up_to_multiple(int(2 * features / 3) * multiplier, 128)``.
For ``features=6144, multiplier=4`` this is ``16384`` (already 128-aligned).
"""

import mlx.core as mx
from mlx import nn


class Krea2SwiGLU(nn.Module):
    def __init__(self, features: int, multiplier: int, bias: bool = False, multiple: int = 128):
        super().__init__()
        mlpdim = int(2 * features / 3) * multiplier
        mlpdim = multiple * ((mlpdim + multiple - 1) // multiple)
        self.gate = nn.Linear(features, mlpdim, bias=bias)
        self.up = nn.Linear(features, mlpdim, bias=bias)
        self.down = nn.Linear(mlpdim, features, bias=bias)

    def __call__(self, x: mx.array) -> mx.array:
        return self.down(nn.silu(self.gate(x)) * self.up(x))
