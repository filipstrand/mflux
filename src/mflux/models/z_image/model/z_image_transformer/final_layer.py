import mlx.core as mx
from mlx import nn


class FinalLayer(nn.Module):
    """Final output layer with AdaLN modulation for Z-Image transformer.

    Z-Image uses scale-only AdaLN (no shift), consistent with its S3-DiT architecture.
    This differs from standard AdaLN-Zero which uses both scale and shift.

    The modulation formula is: x_out = norm(x) * (1 + scale)
    where scale is computed from the conditioning embedding c.
    """

    def __init__(self, hidden_size: int, out_channels: int):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_size, eps=1e-6, affine=False)
        self.linear = nn.Linear(hidden_size, out_channels, bias=True)
        self.adaLN_modulation = [nn.Linear(min(hidden_size, 256), hidden_size, bias=True)]

    def __call__(self, x: mx.array, c: mx.array) -> mx.array:
        scale = 1.0 + self.adaLN_modulation[0](nn.silu(c))
        scale = mx.expand_dims(scale, axis=1)
        x = self.norm(x) * scale
        return self.linear(x)
