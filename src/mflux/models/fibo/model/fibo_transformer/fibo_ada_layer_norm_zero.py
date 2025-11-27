import mlx.core as mx
from mlx import nn


class FiboAdaLayerNormZero(nn.Module):
    def __init__(self, embedding_dim: int, eps: float = 1e-6):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.eps = eps
        self.linear = nn.Linear(embedding_dim, 6 * embedding_dim)

    def __call__(self, hidden_states: mx.array, text_embeddings: mx.array):
        emb = self.linear(nn.silu(text_embeddings))
        chunk = self.embedding_dim
        shift_msa = emb[:, 0 * chunk : 1 * chunk]
        scale_msa = emb[:, 1 * chunk : 2 * chunk]
        gate_msa = emb[:, 2 * chunk : 3 * chunk]
        shift_mlp = emb[:, 3 * chunk : 4 * chunk]
        scale_mlp = emb[:, 4 * chunk : 5 * chunk]
        gate_mlp = emb[:, 5 * chunk : 6 * chunk]
        norm_hidden_states = FiboAdaLayerNormZero._layer_norm(self.eps, hidden_states)
        hidden_states = norm_hidden_states * (1 + scale_msa[:, None, :]) + shift_msa[:, None, :]
        return hidden_states, gate_msa, shift_mlp, scale_mlp, gate_mlp

    @staticmethod
    def _layer_norm(eps, x: mx.array) -> mx.array:
        x_f32 = x.astype(mx.float32)
        mean = mx.mean(x_f32, axis=-1, keepdims=True)
        var = mx.mean((x_f32 - mean) ** 2, axis=-1, keepdims=True)
        y = (x_f32 - mean) / mx.sqrt(var + eps)
        return y.astype(x.dtype)
