import mlx.core as mx
from mlx import nn

from mflux.models.fibo.model.fibo_transformer.feed_forward import FiboFeedForward
from mflux.models.fibo.model.fibo_transformer.fibo_ada_layer_norm_zero import FiboAdaLayerNormZero
from mflux.models.fibo.model.fibo_transformer.fibo_joint_attention import FiboJointAttention


class FiboJointTransformerBlock(nn.Module):
    def __init__(
        self,
        layer: int,
        dim: int = 3072,
        num_attention_heads: int = 24,
        attention_head_dim: int = 128,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.layer = layer
        self.norm1 = FiboAdaLayerNormZero(embedding_dim=dim)
        self.norm1_context = FiboAdaLayerNormZero(embedding_dim=dim)
        self.attn = FiboJointAttention(dim=dim, num_attention_heads=num_attention_heads, attention_head_dim=attention_head_dim)  # fmt: off
        self.norm2 = nn.LayerNorm(dims=dim, eps=eps, affine=False)
        self.ff = FiboFeedForward(dim=dim, dim_out=dim, mult=4, activation_fn="gelu-approximate")
        self.norm2_context = nn.LayerNorm(dims=dim, eps=eps, affine=False)
        self.ff_context = FiboFeedForward(dim=dim, dim_out=dim, mult=4, activation_fn="gelu-approximate")

    def __call__(
        self,
        temb: mx.array,
        hidden_states: mx.array,
        encoder_hidden_states: mx.array,
        image_rotary_emb: mx.array,
        attention_mask: mx.array | None = None,
    ) -> tuple[mx.array, mx.array]:
        # 1. AdaLayerNormZero for both streams
        norm_hidden_states, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.norm1(
            hidden_states=hidden_states,
            text_embeddings=temb,
        )

        norm_encoder_hidden_states, c_gate_msa, c_shift_mlp, c_scale_mlp, c_gate_mlp = self.norm1_context(
            hidden_states=encoder_hidden_states,
            text_embeddings=temb,
        )

        # 2. Joint attention over image + context streams.
        attn_output, context_attn_output = self.attn(
            hidden_states=norm_hidden_states,
            encoder_hidden_states=norm_encoder_hidden_states,
            image_rotary_emb=image_rotary_emb,
            attention_mask=attention_mask,
        )

        # 3a. Process attention outputs for the image stream.
        attn_output = mx.expand_dims(gate_msa, axis=1) * attn_output
        hidden_states = hidden_states + attn_output

        norm_hidden_states = self.norm2(hidden_states)
        norm_hidden_states = norm_hidden_states * (1 + scale_mlp[:, None]) + shift_mlp[:, None]

        ff_output = self.ff(norm_hidden_states)
        ff_output = mx.expand_dims(gate_mlp, axis=1) * ff_output

        hidden_states = hidden_states + ff_output

        # 3b. Process attention outputs for the context stream.
        context_attn_output = mx.expand_dims(c_gate_msa, axis=1) * context_attn_output
        encoder_hidden_states = encoder_hidden_states + context_attn_output

        norm_encoder_hidden_states = self.norm2_context(encoder_hidden_states)
        norm_encoder_hidden_states = norm_encoder_hidden_states * (1 + c_scale_mlp[:, None]) + c_shift_mlp[:, None]

        context_ff_output = self.ff_context(norm_encoder_hidden_states)
        encoder_hidden_states = encoder_hidden_states + mx.expand_dims(c_gate_mlp, axis=1) * context_ff_output
        return encoder_hidden_states, hidden_states
