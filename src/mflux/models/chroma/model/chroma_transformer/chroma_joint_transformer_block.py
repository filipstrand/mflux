"""
Chroma Joint Transformer Block.

This is similar to FLUX's JointTransformerBlock but uses pre-computed modulations
from DistilledGuidanceLayer instead of computing them via linear layers.
"""

import mlx.core as mx
from mlx import nn

from mflux.models.chroma.model.chroma_transformer.chroma_ada_layer_norm import ChromaAdaLayerNormZeroPruned
from mflux.models.flux.model.flux_transformer.feed_forward import FeedForward
from mflux.models.flux.model.flux_transformer.joint_attention import JointAttention


class ChromaJointTransformerBlock(nn.Module):
    """
    Chroma Joint Transformer Block.

    Key differences from FLUX:
    - No norm1.linear or norm1_context.linear weights (modulations pre-computed)
    - Takes 12 modulation vectors directly (6 for image, 6 for text)
    """

    def __init__(self, layer: int, dim: int = 3072):
        super().__init__()
        self.layer = layer
        self.dim = dim

        # Adaptive layer norms without linear projection
        self.norm1 = ChromaAdaLayerNormZeroPruned(dim=dim)
        self.norm1_context = ChromaAdaLayerNormZeroPruned(dim=dim)

        # Joint attention (same as FLUX)
        self.attn = JointAttention()

        # Second layer norms for feed-forward (same as FLUX)
        self.norm2 = nn.LayerNorm(dims=dim, eps=1e-6, affine=False)
        self.norm2_context = nn.LayerNorm(dims=1536, eps=1e-6, affine=False)

        # Feed-forward layers (same as FLUX)
        self.ff = FeedForward(activation_function=nn.gelu)
        self.ff_context = FeedForward(activation_function=nn.gelu_approx)

    def __call__(
        self,
        hidden_states: mx.array,
        encoder_hidden_states: mx.array,
        image_modulations: mx.array,
        text_modulations: mx.array,
        rotary_embeddings: mx.array,
    ) -> tuple[mx.array, mx.array]:
        """
        Forward pass of joint transformer block.

        Args:
            hidden_states: Image hidden states [batch, img_seq_len, dim]
            encoder_hidden_states: Text hidden states [batch, txt_seq_len, dim]
            image_modulations: Pre-computed image modulations [batch, 6, dim]
            text_modulations: Pre-computed text modulations [batch, 6, dim]
            rotary_embeddings: Rotary position embeddings

        Returns:
            Tuple of (updated_encoder_hidden_states, updated_hidden_states)
        """
        # 1a. Compute norm for hidden_states (image)
        norm_hidden_states, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.norm1(
            hidden_states=hidden_states,
            modulations=image_modulations,
        )

        # 1b. Compute norm for encoder_hidden_states (text)
        norm_encoder_hidden_states, c_gate_msa, c_shift_mlp, c_scale_mlp, c_gate_mlp = self.norm1_context(
            hidden_states=encoder_hidden_states,
            modulations=text_modulations,
        )

        # 2. Compute joint attention
        attn_output, context_attn_output = self.attn(
            hidden_states=norm_hidden_states,
            encoder_hidden_states=norm_encoder_hidden_states,
            image_rotary_emb=rotary_embeddings,
        )

        # 3a. Apply norm and feed forward for hidden states (image)
        hidden_states = self._apply_norm_and_feed_forward(
            hidden_states=hidden_states,
            attn_output=attn_output,
            gate_mlp=gate_mlp,
            gate_msa=gate_msa,
            scale_mlp=scale_mlp,
            shift_mlp=shift_mlp,
            norm_layer=self.norm2,
            ff_layer=self.ff,
        )

        # 3b. Apply norm and feed forward for encoder hidden states (text)
        encoder_hidden_states = self._apply_norm_and_feed_forward(
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
    def _apply_norm_and_feed_forward(
        hidden_states: mx.array,
        attn_output: mx.array,
        gate_mlp: mx.array,
        gate_msa: mx.array,
        scale_mlp: mx.array,
        shift_mlp: mx.array,
        norm_layer: nn.Module,
        ff_layer: nn.Module,
    ) -> mx.array:
        """Apply gated attention and feed-forward."""
        attn_output = mx.expand_dims(gate_msa, axis=1) * attn_output
        hidden_states = hidden_states + attn_output
        norm_hidden_states = norm_layer(hidden_states)
        norm_hidden_states = norm_hidden_states * (1 + scale_mlp[:, None]) + shift_mlp[:, None]
        ff_output = ff_layer(norm_hidden_states)
        ff_output = mx.expand_dims(gate_mlp, axis=1) * ff_output
        hidden_states = hidden_states + ff_output
        return hidden_states
