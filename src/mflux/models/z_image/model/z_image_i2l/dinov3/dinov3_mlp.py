import mlx.core as mx
import mlx.nn as nn


class DINOv3GatedMLP(nn.Module):
    """DINOv3 gated MLP with SiLU activation.

    hidden_size=4096, intermediate_size=8192. All projections have bias.
    Formula: down_proj(silu(gate_proj(x)) * up_proj(x))
    """

    def __init__(self):
        super().__init__()
        self.gate_proj = nn.Linear(4096, 8192, bias=True)
        self.up_proj = nn.Linear(4096, 8192, bias=True)
        self.down_proj = nn.Linear(8192, 4096, bias=True)

    def __call__(self, x: mx.array) -> mx.array:
        return self.down_proj(nn.silu(self.gate_proj(x)) * self.up_proj(x))
