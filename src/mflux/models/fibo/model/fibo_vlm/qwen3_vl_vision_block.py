"""Qwen3VL Vision Block for MLX."""

import mlx.core as mx
from mlx import nn

from .qwen3_vl_vision_attention import Qwen3VLVisionAttention
from .qwen3_vl_vision_mlp import Qwen3VLVisionMLP


class Qwen3VLVisionBlock(nn.Module):
    """Transformer block for Qwen3VL vision transformer."""

    def __init__(
        self,
        hidden_size: int = 1024,
        num_heads: int = 16,
        intermediate_size: int = 4096,
        hidden_act: str = "gelu_pytorch_tanh",
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, eps=1e-6)
        self.norm2 = nn.LayerNorm(hidden_size, eps=1e-6)
        self.attn = Qwen3VLVisionAttention(hidden_size=hidden_size, num_heads=num_heads)
        self.mlp = Qwen3VLVisionMLP(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            hidden_act=hidden_act,
        )

    def __call__(
        self,
        hidden_states: mx.array,
        cu_seqlens: mx.array,
        rotary_pos_emb: mx.array | None = None,
        position_embeddings: tuple[mx.array, mx.array] | None = None,
    ) -> mx.array:
        """Forward pass."""
        # Self-attention with residual
        hidden_states = hidden_states + self.attn(
            self.norm1(hidden_states),
            cu_seqlens=cu_seqlens,
            rotary_pos_emb=rotary_pos_emb,
            position_embeddings=position_embeddings,
        )
        # MLP with residual
        hidden_states = hidden_states + self.mlp(self.norm2(hidden_states))
        return hidden_states
