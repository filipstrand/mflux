import mlx.core as mx
from mlx import nn


class QwenTransformerRMSNorm(nn.Module):
    """
    RMSNorm that matches PyTorch's RMSNorm implementation exactly (diffusers/models/normalization.py:511-568).
    Only variance calculation uses float32, main computation stays in original dtype.
    """

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        # PyTorch: elementwise_affine defaults to True, so we create weight parameter
        # PyTorch: self.weight = nn.Parameter(torch.ones(dim))
        self.weight = mx.ones((dim,))
        self.eps = eps

    def __call__(self, hidden_states: mx.array) -> mx.array:
        """
        Matches PyTorch RMSNorm.forward exactly (lines 554-566).

        PyTorch:
            input_dtype = hidden_states.dtype
            variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
            hidden_states = hidden_states * torch.rsqrt(variance + self.eps)
            if self.weight is not None:
                if self.weight.dtype in [torch.float16, torch.bfloat16]:
                    hidden_states = hidden_states.to(self.weight.dtype)
                hidden_states = hidden_states * self.weight
        """
        # PyTorch: input_dtype = hidden_states.dtype
        input_dtype = hidden_states.dtype

        # PyTorch: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        # Only variance calculation uses float32
        variance = mx.power(hidden_states.astype(mx.float32), 2).mean(axis=-1, keepdims=True)

        # PyTorch: hidden_states = hidden_states * torch.rsqrt(variance + self.eps)
        # Main computation stays in original dtype
        hidden_states = hidden_states * mx.rsqrt(variance + self.eps)

        # PyTorch: if self.weight is not None:
        if self.weight is not None:
            # PyTorch: if self.weight.dtype in [torch.float16, torch.bfloat16]:
            #          hidden_states = hidden_states.to(self.weight.dtype)
            # Match PyTorch: convert to weight dtype if half-precision
            if self.weight.dtype in [mx.bfloat16, mx.float16]:
                hidden_states = hidden_states.astype(self.weight.dtype)
            # PyTorch: hidden_states = hidden_states * self.weight
            hidden_states = hidden_states * self.weight
            # Match PyTorch: convert back to input dtype if needed
            if hidden_states.dtype != input_dtype:
                hidden_states = hidden_states.astype(input_dtype)

        return hidden_states
