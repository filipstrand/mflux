import mlx.core as mx
from mlx import nn


class IrisModulationBias(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.bias = mx.zeros((6 * dim,))

    def __call__(self, core_out: mx.array) -> mx.array:
        return core_out + self.bias
