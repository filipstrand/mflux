from mlx import nn
import mlx.core as mx

from mflux.config.config import Config


class AdaLayerNormContinuous(nn.Module):

    def __init__(self, embedding_dim: int, conditioning_embedding_dim: int):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.linear = nn.Linear(conditioning_embedding_dim, embedding_dim * 2)
        self.norm = nn.LayerNorm(dims=embedding_dim, eps=1e-6, affine=False)

    def forward(self, x: mx.array, text_embeddings: mx.array) -> mx.array:
        text_embeddings = self.linear(nn.silu(text_embeddings).astype(Config.precision))
        chunk_size = self.embedding_dim
        scale = text_embeddings[:, 0*chunk_size:1*chunk_size]
        shift = text_embeddings[:, 1*chunk_size:2*chunk_size]
        x = self.norm(x) * (1 + scale)[:, None, :] + shift[:, None, :]
        return x

