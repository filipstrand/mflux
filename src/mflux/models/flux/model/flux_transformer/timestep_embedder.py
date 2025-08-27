import mlx.core as mx
from mlx import nn


class TimestepEmbedder(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_1 = nn.Linear(256, 3072)
        self.linear_2 = nn.Linear(3072, 3072)

    def __call__(self, sample: mx.array) -> mx.array:
        sample = self.linear_1(sample)
        sample = nn.silu(sample)
        sample = self.linear_2(sample)
        return sample
