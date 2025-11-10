import mlx.core as mx
from mlx import nn

from mflux.models.qwen.model.qwen_text_encoder.qwen_vision_attention import VisionAttention
from mflux.models.qwen.model.qwen_text_encoder.qwen_vision_mlp import VisionMLP


class VisionBlock(nn.Module):
    def __init__(self, embed_dim: int = 1280, num_heads: int = 16, mlp_ratio: float = 2.671875):
        super().__init__()
        self.norm1 = nn.RMSNorm(embed_dim, eps=1e-6)  # Fixed: was LayerNorm, should be RMSNorm
        self.norm2 = nn.RMSNorm(embed_dim, eps=1e-6)  # Fixed: was LayerNorm, should be RMSNorm
        self.attn = VisionAttention(embed_dim, num_heads)
        mlp_hidden_dim = int(embed_dim * mlp_ratio)  # 3420 for 1280 * 2.671875
        self.mlp = VisionMLP(embed_dim, mlp_hidden_dim)

    def __call__(self, x: mx.array, position_embeddings=None, cu_seqlens=None) -> mx.array:
        # Pre-norm architecture
        # TEMP DEBUG: Track divergence within block (expanded for breakpoint tracing)
        normed1 = self.norm1(x)
        attn_out = self.attn(normed1, position_embeddings, cu_seqlens)
        x = x + attn_out

        normed2 = self.norm2(x)
        mlp_out = self.mlp(normed2)
        x = x + mlp_out
        return x
