import mlx.core as mx
from mlx import nn

from mflux.models.qwen.model.qwen_text_encoder.qwen_vision_attention import VisionAttention
from mflux.models.qwen.model.qwen_text_encoder.qwen_vision_mlp import VisionMLP


class VisionBlock(nn.Module):
    def __init__(self, embed_dim: int = 1280, num_heads: int = 16, mlp_ratio: float = 2.671875):
        super().__init__()
        self.norm1 = nn.RMSNorm(embed_dim, eps=1e-6)
        self.norm2 = nn.RMSNorm(embed_dim, eps=1e-6)
        self.attn = VisionAttention(embed_dim, num_heads)
        mlp_hidden_dim = int(embed_dim * mlp_ratio)
        self.mlp = VisionMLP(embed_dim, mlp_hidden_dim)

    def __call__(self, x: mx.array, position_embeddings=None, cu_seqlens=None) -> mx.array:
        normed1 = self.norm1(x)
        attn_out = self.attn(normed1, position_embeddings, cu_seqlens)
        x = x + attn_out
        normed2 = self.norm2(x)
        mlp_out = self.mlp(normed2)
        x = x + mlp_out
        return x
