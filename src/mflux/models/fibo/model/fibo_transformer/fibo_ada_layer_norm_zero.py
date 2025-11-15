import mlx.core as mx
from mlx import nn


class FiboAdaLayerNormZero(nn.Module):
    """
    FIBO-specific AdaLayerNormZero that mirrors the PyTorch
    `diffusers.models.normalization.AdaLayerNormZero` behavior:

        - SiLU + linear(embedding_dim -> 6 * embedding_dim)
        - Chunk into (shift_msa, scale_msa, gate_msa,
                      shift_mlp, scale_mlp, gate_mlp)
        - LayerNorm over the last dim of `hidden_states`
        - Modulation: norm * (1 + scale_msa) + shift_msa

    We implement LayerNorm explicitly to avoid subtle numerical
    differences with `nn.LayerNorm` and to match PyTorch exactly.
    """

    def __init__(self, embedding_dim: int, eps: float = 1e-6):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.eps = eps

        # Matches diffusers: nn.Linear(embedding_dim, 6 * embedding_dim, bias=True)
        self.linear = nn.Linear(embedding_dim, 6 * embedding_dim)

    def _layer_norm(self, x: mx.array) -> mx.array:
        """
        PyTorch-style LayerNorm over the last dimension, with no affine params.
        """
        x_f32 = x.astype(mx.float32)
        mean = mx.mean(x_f32, axis=-1, keepdims=True)
        var = mx.mean((x_f32 - mean) ** 2, axis=-1, keepdims=True)
        y = (x_f32 - mean) / mx.sqrt(var + self.eps)
        return y.astype(x.dtype)

    def __call__(self, hidden_states: mx.array, text_embeddings: mx.array):
        # text_embeddings: (B, embedding_dim) == temb
        emb = self.linear(nn.silu(text_embeddings))
        chunk = self.embedding_dim

        shift_msa = emb[:, 0 * chunk : 1 * chunk]
        scale_msa = emb[:, 1 * chunk : 2 * chunk]
        gate_msa = emb[:, 2 * chunk : 3 * chunk]
        shift_mlp = emb[:, 3 * chunk : 4 * chunk]
        scale_mlp = emb[:, 4 * chunk : 5 * chunk]
        gate_mlp = emb[:, 5 * chunk : 6 * chunk]

        # LayerNorm over last dim of hidden_states: (B, seq, embedding_dim)
        norm_hidden_states = self._layer_norm(hidden_states)

        # Broadcast shift/scale over sequence dimension
        hidden_states = norm_hidden_states * (1 + scale_msa[:, None, :]) + shift_msa[:, None, :]

        return hidden_states, gate_msa, shift_mlp, scale_mlp, gate_mlp
