import mlx.core as mx
from mlx import nn

from mflux.models.iris.model.iris_transformer.iris_rms_norm import IrisRMSNorm


class IrisFinalLayer(nn.Module):
    def __init__(self, hidden_size: int, out_dim: int, eps: float = 1e-6):
        super().__init__()
        self.norm = IrisRMSNorm(hidden_size, eps)
        self.linear = nn.Linear(hidden_size, out_dim, bias=True)

    def __call__(self, x: mx.array) -> mx.array:
        return self.linear(self.norm(x))
