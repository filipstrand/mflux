import mlx.core as mx
import numpy as np
from mlx import nn


class Qwen3VLVisionMLP(nn.Module):
    def __init__(
        self,
        hidden_size: int = 1024,
        intermediate_size: int = 4096,
        hidden_act: str = "gelu_pytorch_tanh",
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.linear_fc1 = nn.Linear(hidden_size, intermediate_size, bias=True)
        self.linear_fc2 = nn.Linear(intermediate_size, hidden_size, bias=True)

        if hidden_act == "gelu_pytorch_tanh":
            self.act_fn = lambda x: 0.5 * x * (1 + mx.tanh(mx.sqrt(2 / np.pi) * (x + 0.044715 * x**3)))
        elif hidden_act == "gelu":
            self.act_fn = lambda x: x * 0.5 * (1.0 + mx.erf(x / mx.sqrt(2.0)))
        else:
            raise ValueError(f"Unsupported activation: {hidden_act}")

    def __call__(self, hidden_state: mx.array) -> mx.array:
        return self.linear_fc2(self.act_fn(self.linear_fc1(hidden_state)))
