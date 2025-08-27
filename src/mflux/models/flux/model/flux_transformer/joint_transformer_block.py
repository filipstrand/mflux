import mlx.core as mx
from mlx import nn

from mflux.models.flux.model.flux_transformer.ada_layer_norm_zero import AdaLayerNormZero
from mflux.models.flux.model.flux_transformer.feed_forward import FeedForward
from mflux.models.flux.model.flux_transformer.joint_attention import JointAttention


class JointTransformerBlock(nn.Module):
    def __init__(self, layer):
        super().__init__()
        self.layer = layer
        self.norm1 = AdaLayerNormZero()
        self.norm1_context = AdaLayerNormZero()
        self.attn = JointAttention()
        self.norm2 = nn.LayerNorm(dims=3072, eps=1e-6, affine=False)
        self.norm2_context = nn.LayerNorm(dims=1536, eps=1e-6, affine=False)
        self.ff = FeedForward(activation_function=nn.gelu)
        self.ff_context = FeedForward(activation_function=nn.gelu_approx)

    def __call__(
        self,
        hidden_states: mx.array,
        encoder_hidden_states: mx.array,
        text_embeddings: mx.array,
        rotary_embeddings: mx.array,
    ) -> tuple[mx.array, mx.array]:
        # 1a. Compute norm for hidden_states
        norm_hidden_states, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.norm1(
            hidden_states=hidden_states,
            text_embeddings=text_embeddings,
        )

        # 1b. Compute norm for encoder_hidden_states
        norm_encoder_hidden_states, c_gate_msa, c_shift_mlp, c_scale_mlp, c_gate_mlp = self.norm1_context(
            hidden_states=encoder_hidden_states,
            text_embeddings=text_embeddings,
        )

        # 2. Compute attention
        attn_output, context_attn_output = self.attn(
            hidden_states=norm_hidden_states,
            encoder_hidden_states=norm_encoder_hidden_states,
            image_rotary_emb=rotary_embeddings,
        )

        # 3a. Apply norm and feed forward for hidden states
        hidden_states = JointTransformerBlock.apply_norm_and_feed_forward(
            hidden_states=hidden_states,
            attn_output=attn_output,
            gate_mlp=gate_mlp,
            gate_msa=gate_msa,
            scale_mlp=scale_mlp,
            shift_mlp=shift_mlp,
            norm_layer=self.norm2,
            ff_layer=self.ff,
        )

        # 3b. Apply norm and feed forward for encoder hidden states
        encoder_hidden_states = JointTransformerBlock.apply_norm_and_feed_forward(
            hidden_states=encoder_hidden_states,
            attn_output=context_attn_output,
            gate_mlp=c_gate_mlp,
            gate_msa=c_gate_msa,
            scale_mlp=c_scale_mlp,
            shift_mlp=c_shift_mlp,
            norm_layer=self.norm2_context,
            ff_layer=self.ff_context,
        )

        return encoder_hidden_states, hidden_states

    @staticmethod
    def apply_norm_and_feed_forward(
        hidden_states: mx.array,
        attn_output: mx.array,
        gate_mlp: mx.array,
        gate_msa: mx.array,
        scale_mlp: mx.array,
        shift_mlp: mx.array,
        norm_layer: nn.Module,
        ff_layer: nn.Module,
    ) -> mx.array:
        attn_output = mx.expand_dims(gate_msa, axis=1) * attn_output
        hidden_states = hidden_states + attn_output
        norm_hidden_states = norm_layer(hidden_states)
        norm_hidden_states = norm_hidden_states * (1 + scale_mlp[:, None]) + shift_mlp[:, None]
        ff_output = ff_layer(norm_hidden_states)
        ff_output = mx.expand_dims(gate_mlp, axis=1) * ff_output
        hidden_states = hidden_states + ff_output
        return hidden_states
