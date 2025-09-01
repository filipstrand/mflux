import mlx.core as mx
from mlx import nn


class ReduxEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.redux_up = nn.Linear(1152, 4096 * 3)
        self.redux_down = nn.Linear(4096 * 3, 4096)

    def __call__(self, x: mx.array) -> mx.array:
        return self.redux_down(nn.silu(self.redux_up(x)))
