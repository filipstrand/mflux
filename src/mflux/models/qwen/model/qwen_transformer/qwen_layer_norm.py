import mlx.core as mx
from mlx import nn


class QwenLayerNorm(nn.Module):
    def __init__(self, dim: int = 3072):
        super().__init__()
        self.mod_linear = nn.Linear(dim, 6 * dim)
        self.norm1 = nn.LayerNorm(dims=dim, eps=1e-6, affine=False)

    def __call__(self, hidden_states: mx.array, text_embeddings: mx.array) -> tuple[mx.array, mx.array, mx.array]:
        temb_silu = nn.silu(text_embeddings)
        mod_params = self.mod_linear(temb_silu)
        mod1, mod2 = mx.split(mod_params, 2, axis=-1)
        normed1 = self.norm1(hidden_states)
        shift1, scale1, gate1 = mx.split(mod1, 3, axis=-1)
        normed_stage1 = normed1 * (1 + scale1[:, None, :]) + shift1[:, None, :]
        return normed_stage1, gate1, mod2
