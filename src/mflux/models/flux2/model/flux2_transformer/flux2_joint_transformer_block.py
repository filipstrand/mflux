"""Joint transformer block for FLUX.2.

FLUX.2 joint blocks are similar to FLUX.1 but:
- No per-block norm layers (modulation comes from global modulation layers)
- Different FFN naming (linear_in/linear_out vs net.0.proj/net.2)
- 48 attention heads (vs 24 in FLUX.1)
- 6144 hidden dim (vs 3072 in FLUX.1)
"""

import mlx.core as mx
from mlx import nn

from mflux.models.flux2.model.flux2_transformer.flux2_feed_forward import (
    Flux2FeedForward,
    Flux2FeedForwardContext,
)
from mflux.models.flux2.model.flux2_transformer.flux2_joint_attention import Flux2JointAttention


class Flux2JointTransformerBlock(nn.Module):
    """Joint transformer block for FLUX.2.

    Processes both image and text/context streams jointly.
    Modulation parameters are provided externally from global modulation layers.
    """

    def __init__(self, layer: int):
        super().__init__()
        self.layer = layer
        self.hidden_dim = 6144

        # Attention
        self.attn = Flux2JointAttention()

        # Layer norms (applied before attention and FFN)
        self.norm1 = nn.LayerNorm(dims=self.hidden_dim, eps=1e-6, affine=False)
        self.norm1_context = nn.LayerNorm(dims=self.hidden_dim, eps=1e-6, affine=False)
        self.norm2 = nn.LayerNorm(dims=self.hidden_dim, eps=1e-6, affine=False)
        self.norm2_context = nn.LayerNorm(dims=self.hidden_dim, eps=1e-6, affine=False)

        # Feed forward networks
        self.ff = Flux2FeedForward(hidden_dim=self.hidden_dim, activation_function=nn.gelu)
        self.ff_context = Flux2FeedForwardContext(hidden_dim=self.hidden_dim, activation_function=nn.gelu_approx)

    def __call__(
        self,
        hidden_states: mx.array,
        encoder_hidden_states: mx.array,
        rotary_embeddings: mx.array,
        img_modulation: tuple[mx.array, ...],
        txt_modulation: tuple[mx.array, ...],
    ) -> tuple[mx.array, mx.array]:
        """Process joint transformer block.

        Args:
            hidden_states: Image hidden states [batch, img_seq, hidden_dim]
            encoder_hidden_states: Text/context hidden states [batch, txt_seq, hidden_dim]
            rotary_embeddings: RoPE embeddings
            img_modulation: (shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp) for image
            txt_modulation: (shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp) for text

        Returns:
            Tuple of (encoder_hidden_states, hidden_states)
        """
        # Unpack modulation parameters
        img_shift_msa, img_scale_msa, img_gate_msa, img_shift_mlp, img_scale_mlp, img_gate_mlp = img_modulation
        txt_shift_msa, txt_scale_msa, txt_gate_msa, txt_shift_mlp, txt_scale_mlp, txt_gate_mlp = txt_modulation

        # 1a. Apply norm and modulation for image hidden_states
        norm_hidden_states = self.norm1(hidden_states)
        norm_hidden_states = norm_hidden_states * (1 + img_scale_msa[:, None]) + img_shift_msa[:, None]

        # 1b. Apply norm and modulation for text encoder_hidden_states
        norm_encoder_hidden_states = self.norm1_context(encoder_hidden_states)
        norm_encoder_hidden_states = norm_encoder_hidden_states * (1 + txt_scale_msa[:, None]) + txt_shift_msa[:, None]

        # 2. Compute joint attention
        attn_output, context_attn_output = self.attn(
            hidden_states=norm_hidden_states,
            encoder_hidden_states=norm_encoder_hidden_states,
            image_rotary_emb=rotary_embeddings,
        )

        # 3a. Apply gating and residual for image stream
        attn_output = mx.expand_dims(img_gate_msa, axis=1) * attn_output
        hidden_states = hidden_states + attn_output

        # 3b. Apply gating and residual for text stream
        context_attn_output = mx.expand_dims(txt_gate_msa, axis=1) * context_attn_output
        encoder_hidden_states = encoder_hidden_states + context_attn_output

        # 4a. FFN for image stream
        norm_hidden_states = self.norm2(hidden_states)
        norm_hidden_states = norm_hidden_states * (1 + img_scale_mlp[:, None]) + img_shift_mlp[:, None]
        ff_output = self.ff(norm_hidden_states)
        ff_output = mx.expand_dims(img_gate_mlp, axis=1) * ff_output
        hidden_states = hidden_states + ff_output

        # 4b. FFN for text stream
        norm_encoder_hidden_states = self.norm2_context(encoder_hidden_states)
        norm_encoder_hidden_states = norm_encoder_hidden_states * (1 + txt_scale_mlp[:, None]) + txt_shift_mlp[:, None]
        ff_context_output = self.ff_context(norm_encoder_hidden_states)
        ff_context_output = mx.expand_dims(txt_gate_mlp, axis=1) * ff_context_output
        encoder_hidden_states = encoder_hidden_states + ff_context_output

        return encoder_hidden_states, hidden_states
