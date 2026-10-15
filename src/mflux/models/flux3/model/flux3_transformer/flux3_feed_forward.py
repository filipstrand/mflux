import mlx.core as mx
from mlx import nn

from mflux.models.common.compute_precision import ComputePrecision


class Flux3FeedForward(nn.Module):
    # The output reaches ~3e5, beyond float16: the up half is taken this many times smaller and the output
    # scaled back in the stream dtype (exact up to rounding).
    FLOAT16_HEADROOM = 8.0

    def __init__(self, dim: int, mlp_ratio: float = 3.0):
        super().__init__()
        hidden = int(dim * mlp_ratio)
        self.linear_in = nn.Linear(dim, hidden * 2, bias=False)
        self.linear_out = nn.Linear(hidden, dim, bias=False)
        self.compute_precision = ComputePrecision()

    def __call__(self, x: mx.array) -> mx.array:
        precision = self.compute_precision
        dtype = x.dtype
        gate, up = mx.split(self.linear_in(precision.to_compute(x)), 2, axis=-1)
        hidden = nn.silu(gate) * precision.shrink(up, Flux3FeedForward.FLOAT16_HEADROOM)
        return precision.to_stream(self.linear_out(hidden), dtype, Flux3FeedForward.FLOAT16_HEADROOM)
