from mlx import nn
import mlx.core as mx


class AdaLayerNormZero(nn.Module):

    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(3072, 18432)
        self.norm = nn.LayerNorm(dims=3072, eps=1e-6, affine=False)

    def forward(self, x: mx.array, text_embeddings: mx.array):
        text_embeddings = self.linear(nn.silu(text_embeddings))
        chunk_size = 18432 // 6
        shift_msa = text_embeddings[:, 0*chunk_size:1*chunk_size]
        scale_msa = text_embeddings[:, 1*chunk_size:2*chunk_size]
        gate_msa = text_embeddings[:, 2*chunk_size:3*chunk_size]
        shift_mlp = text_embeddings[:, 3*chunk_size:4*chunk_size]
        scale_mlp = text_embeddings[:, 4*chunk_size:5*chunk_size]
        gate_mlp = text_embeddings[:, 5*chunk_size:6*chunk_size]
        x = self.norm(x) * (1 + scale_msa[:, None]) + shift_msa[:, None]
        return x, gate_msa, shift_mlp, scale_mlp, gate_mlp
