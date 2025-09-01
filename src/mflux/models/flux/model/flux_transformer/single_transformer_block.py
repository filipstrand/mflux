import mlx.core as mx
from mlx import nn

from mflux.models.flux.model.flux_transformer.ada_layer_norm_zero_single import AdaLayerNormZeroSingle
from mflux.models.flux.model.flux_transformer.single_block_attention import SingleBlockAttention


class SingleTransformerBlock(nn.Module):
    def __init__(self, layer):
        super().__init__()
        self.layer = layer
        self.norm = AdaLayerNormZeroSingle()
        self.attn = SingleBlockAttention()
        self.proj_mlp = nn.Linear(3072, 4 * 3072)
        self.proj_out = nn.Linear(3072 + 4 * 3072, 3072)

    def __call__(
        self,
        hidden_states: mx.array,
        text_embeddings: mx.array,
        rotary_embeddings: mx.array,
    ) -> tuple[mx.array, mx.array]:
        # 0. Establish residual connection
        residual = hidden_states

        # 1. Compute norm for hidden_states
        norm_hidden_states, gate = self.norm(
            hidden_states=hidden_states,
            text_embeddings=text_embeddings,
        )

        # 2. Compute attention
        attn_output = self.attn(
            hidden_states=norm_hidden_states,
            image_rotary_emb=rotary_embeddings,
        )

        # 3. Apply norm and feed forward for hidden states
        hidden_states = self._apply_feed_forward_and_projection(
            norm_hidden_states=norm_hidden_states,
            attn_output=attn_output,
            gate=gate,
        )

        return residual + hidden_states

    def _apply_feed_forward_and_projection(
        self,
        norm_hidden_states: mx.array,
        attn_output: mx.array,
        gate: mx.array,
    ) -> mx.array:
        feed_forward = self.proj_mlp(norm_hidden_states)
        mlp_hidden_states = nn.gelu_approx(feed_forward)
        hidden_states = mx.concatenate([attn_output, mlp_hidden_states], axis=2)
        gate = mx.expand_dims(gate, axis=1)
        hidden_states = gate * self.proj_out(hidden_states)
        return hidden_states
