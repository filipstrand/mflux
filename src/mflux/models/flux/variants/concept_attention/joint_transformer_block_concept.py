from dataclasses import dataclass

import mlx.core as mx
from mlx import nn

from mflux.models.flux.model.flux_transformer.ada_layer_norm_zero import AdaLayerNormZero
from mflux.models.flux.model.flux_transformer.feed_forward import FeedForward
from mflux.models.flux.model.flux_transformer.joint_transformer_block import JointTransformerBlock
from mflux.models.flux.variants.concept_attention.joint_attention_concept import JointAttentionConcept


@dataclass
class LayerAttentionData:
    layer: int
    img_attention: mx.array
    concept_attention: mx.array


class JointTransformerBlockConcept(nn.Module):
    def __init__(self, layer):
        super().__init__()
        self.layer = layer
        self.norm1 = AdaLayerNormZero()
        self.norm1_context = AdaLayerNormZero()
        self.attn = JointAttentionConcept()
        self.norm2 = nn.LayerNorm(dims=3072, eps=1e-6, affine=False)
        self.norm2_context = nn.LayerNorm(dims=1536, eps=1e-6, affine=False)
        self.ff = FeedForward(activation_function=nn.gelu)
        self.ff_context = FeedForward(activation_function=nn.gelu_approx)

    def __call__(
        self,
        layer_idx: int,
        hidden_states: mx.array,
        encoder_hidden_states: mx.array,
        encoder_hidden_states_concept: mx.array,
        text_embeddings: mx.array,
        text_embeddings_concept: mx.array,
        rotary_embeddings: mx.array,
        rotary_embeddings_concept: mx.array,
    ) -> tuple[mx.array, mx.array, mx.array, LayerAttentionData]:
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

        # 1c. Compute norm for encoder_hidden_states_concept
        norm_encoder_hidden_states_concept, c_gate_msa_concept, c_shift_mlp_concept, c_scale_mlp_concept, c_gate_mlp_concept = self.norm1_context(
            hidden_states=encoder_hidden_states_concept,
            text_embeddings=text_embeddings_concept,
        )  # fmt: off

        # 2. Compute attention
        attn_output, context_attn_output, context_attn_output_concept, img_attn, concept_attn = self.attn(
            hidden_states=norm_hidden_states,
            encoder_hidden_states=norm_encoder_hidden_states,
            encoder_hidden_states_concept=norm_encoder_hidden_states_concept,
            image_rotary_emb=rotary_embeddings,
            image_rotary_emb_concept=rotary_embeddings_concept,
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

        # 3c. Apply norm and feed forward for concept encoder hidden states
        encoder_hidden_states_concept = JointTransformerBlock.apply_norm_and_feed_forward(
            hidden_states=encoder_hidden_states_concept,
            attn_output=context_attn_output_concept,
            gate_mlp=c_gate_mlp_concept,
            gate_msa=c_gate_msa_concept,
            scale_mlp=c_scale_mlp_concept,
            shift_mlp=c_shift_mlp_concept,
            norm_layer=self.norm2_context,
            ff_layer=self.ff_context,
        )

        # 4. Package attention data with layer information
        layer_attention_data = LayerAttentionData(
            layer=layer_idx,
            img_attention=img_attn,
            concept_attention=concept_attn,
        )

        return encoder_hidden_states, hidden_states, encoder_hidden_states_concept, layer_attention_data
