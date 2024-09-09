import mlx.core as mx
from mlx import nn

from mflux.models.transformer.ada_layer_norm_zero import AdaLayerNormZero
from mflux.models.transformer.feed_forward import FeedForward
from mflux.models.transformer.joint_attention import JointAttention


class JointTransformerBlock(nn.Module):

    def __init__(self, layer):
        super().__init__()
        self.layer = layer
        self.norm1 = AdaLayerNormZero()
        self.norm2 = nn.LayerNorm(dims=3072, eps=1e-6, affine=False)
        self.ff = FeedForward(activation_function=nn.gelu)
        self.attn = JointAttention()
        self.norm1_context = AdaLayerNormZero()
        self.ff_context = FeedForward(activation_function=nn.gelu_approx)
        self.norm2_context = nn.LayerNorm(dims=1536, eps=1e-6, affine=False)

    def forward(
            self,
            hidden_states: mx.array,
            encoder_hidden_states: mx.array,
            text_embeddings: mx.array,
            rotary_embeddings: mx.array
    ) -> (mx.array, mx.array):
        norm_hidden_states, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.norm1.forward(hidden_states, text_embeddings)

        norm_encoder_hidden_states, c_gate_msa, c_shift_mlp, c_scale_mlp, c_gate_mlp = self.norm1_context.forward(
            x=encoder_hidden_states,
            text_embeddings=text_embeddings
        )

        attn_output, context_attn_output = self.attn.forward(
            hidden_states=norm_hidden_states,
            encoder_hidden_states=norm_encoder_hidden_states,
            image_rotary_emb=rotary_embeddings,
        )

        attn_output = mx.expand_dims(gate_msa, axis=1) * attn_output
        hidden_states = hidden_states + attn_output
        norm_hidden_states = self.norm2(hidden_states)
        norm_hidden_states = norm_hidden_states * (1 + scale_mlp[:, None]) + shift_mlp[:, None]
        ff_output = self.ff.forward(norm_hidden_states)
        ff_output = mx.expand_dims(gate_mlp, axis=1) * ff_output
        hidden_states = hidden_states + ff_output

        context_attn_output = mx.expand_dims(c_gate_msa, axis=1) * context_attn_output
        encoder_hidden_states = encoder_hidden_states + context_attn_output
        norm_encoder_hidden_states = self.norm2_context(encoder_hidden_states)
        norm_encoder_hidden_states = norm_encoder_hidden_states * (1 + c_scale_mlp[:, None]) + c_shift_mlp[:, None]
        context_ff_output = self.ff_context.forward(norm_encoder_hidden_states)
        encoder_hidden_states = encoder_hidden_states + mx.expand_dims(c_gate_mlp, axis=1) * context_ff_output
        return encoder_hidden_states, hidden_states
