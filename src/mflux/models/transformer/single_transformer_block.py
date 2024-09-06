import mlx.core as mx
from mlx import nn

from mflux.models.transformer.ada_layer_norm_zero_single import AdaLayerNormZeroSingle
from mflux.models.transformer.single_block_attention import SingleBlockAttention


class SingleTransformerBlock(nn.Module):

    def __init__(self, layer):
        super().__init__()
        self.layer = layer
        self.norm = AdaLayerNormZeroSingle()
        self.proj_mlp = nn.Linear(3072, 4*3072)
        self.attn = SingleBlockAttention()
        self.proj_out = nn.Linear(3072 + 4*3072, 3072)

    def forward(
            self,
            hidden_states: mx.array,
            text_embeddings: mx.array,
            rotary_embeddings: mx.array
    ) -> (mx.array, mx.array):
        residual = hidden_states
        norm_hidden_states, gate = self.norm.forward(x=hidden_states, text_embeddings=text_embeddings)
        mlp_hidden_states = nn.gelu_approx(self.proj_mlp(norm_hidden_states))
        attn_output = self.attn.forward(
            hidden_states=norm_hidden_states,
            image_rotary_emb=rotary_embeddings,
        )
        hidden_states = mx.concatenate([attn_output, mlp_hidden_states], axis=2)
        gate = mx.expand_dims(gate, axis=1)
        hidden_states = gate * self.proj_out(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states
