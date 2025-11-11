import mlx.core as mx
from mlx import nn


class QwenTransformerRMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = mx.ones((dim,))
        self.eps = eps

    def __call__(self, hidden_states: mx.array) -> mx.array:
        input_dtype = hidden_states.dtype
        variance = mx.power(hidden_states.astype(mx.float32), 2).mean(axis=-1, keepdims=True)
        hidden_states = hidden_states * mx.rsqrt(variance + self.eps)

        if self.weight is not None:
            if self.weight.dtype in [mx.bfloat16, mx.float16]:
                hidden_states = hidden_states.astype(self.weight.dtype)
            hidden_states = hidden_states * self.weight
            if hidden_states.dtype != input_dtype:
                hidden_states = hidden_states.astype(input_dtype)

        return hidden_states
