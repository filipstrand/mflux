import mlx.core as mx
from mlx import nn

from mflux.models.flux.model.flux_transformer.ada_layer_norm_zero_single import AdaLayerNormZeroSingle
from mflux.models.flux.model.flux_transformer.single_block_attention import SingleBlockAttention


class FiboSingleTransformerBlock(nn.Module):
    """
    MLX port of diffusers.models.transformers.transformer_bria_fibo.BriaFiboSingleTransformerBlock.
    """

    def __init__(self, dim: int, num_attention_heads: int, attention_head_dim: int, mlp_ratio: float = 4.0):
        super().__init__()
        self.mlp_hidden_dim = int(dim * mlp_ratio)

        self.norm = AdaLayerNormZeroSingle()
        self.proj_mlp = nn.Linear(dim, self.mlp_hidden_dim)
        self.act_mlp = nn.gelu_approx
        self.proj_out = nn.Linear(dim + self.mlp_hidden_dim, dim)

        self.attn = SingleBlockAttention()

    def __call__(
        self,
        hidden_states: mx.array,
        temb: mx.array,
        image_rotary_emb: mx.array,
    ) -> mx.array:
        # 0. Residual connection
        residual = hidden_states

        # 1. AdaLayerNormZeroSingle
        norm_hidden_states, gate = self.norm(
            hidden_states=hidden_states,
            text_embeddings=temb,
        )

        # 2. Attention
        attn_output = self.attn(
            hidden_states=norm_hidden_states,
            image_rotary_emb=image_rotary_emb,
        )

        # 3. MLP + projection (mirrors BriaFiboSingleTransformerBlock)
        mlp_hidden_states = self.act_mlp(self.proj_mlp(norm_hidden_states))
        hidden_states = mx.concatenate([attn_output, mlp_hidden_states], axis=2)
        gate = mx.expand_dims(gate, axis=1)
        hidden_states = gate * self.proj_out(hidden_states)

        hidden_states = residual + hidden_states
        if hidden_states.dtype == mx.float16:
            hidden_states = mx.clip(hidden_states, -65504.0, 65504.0)

        return hidden_states
