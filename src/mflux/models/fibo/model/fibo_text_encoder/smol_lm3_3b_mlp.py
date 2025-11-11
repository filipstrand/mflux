import mlx.core as mx
from mlx import nn


class SmolLM3_3B_MLP(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str = "silu",
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def _activation(self, x: mx.array) -> mx.array:
        if self.hidden_act == "silu":
            return x * mx.sigmoid(x)
        return x * mx.sigmoid(x)

    def __call__(self, hidden_states: mx.array) -> mx.array:
        gate = self._activation(self.gate_proj(hidden_states))
        up = self.up_proj(hidden_states)
        hidden = gate * up
        hidden = self.down_proj(hidden)
        return hidden
