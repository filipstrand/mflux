"""
Chroma Single Transformer Block.

This is similar to FLUX's SingleTransformerBlock but uses pre-computed modulations
from DistilledGuidanceLayer instead of computing them via a linear layer.
"""

import mlx.core as mx
from mlx import nn

from mflux.models.chroma.model.chroma_transformer.chroma_ada_layer_norm import ChromaAdaLayerNormZeroSinglePruned
from mflux.models.flux.model.flux_transformer.single_block_attention import SingleBlockAttention


class ChromaSingleTransformerBlock(nn.Module):
    """
    Chroma Single Transformer Block.

    Key difference from FLUX:
    - No norm.linear weight (modulations pre-computed)
    - Takes 3 modulation vectors directly instead of computing from text_embeddings
    """

    def __init__(self, layer: int, dim: int = 3072, mlp_ratio: float = 4.0):
        super().__init__()
        self.layer = layer
        self.dim = dim
        self.mlp_hidden_dim = int(dim * mlp_ratio)

        # Adaptive layer norm without linear projection
        self.norm = ChromaAdaLayerNormZeroSinglePruned(dim=dim)

        # Attention (same as FLUX)
        self.attn = SingleBlockAttention()

        # MLP projections (same as FLUX)
        self.proj_mlp = nn.Linear(dim, self.mlp_hidden_dim)
        self.proj_out = nn.Linear(dim + self.mlp_hidden_dim, dim)

    def __call__(
        self,
        hidden_states: mx.array,
        modulations: mx.array,
        rotary_embeddings: mx.array,
    ) -> mx.array:
        """
        Forward pass of single transformer block.

        Args:
            hidden_states: Input tensor [batch, seq_len, dim]
            modulations: Pre-computed modulations [batch, 3, dim] for (shift, scale, gate)
            rotary_embeddings: Rotary position embeddings

        Returns:
            Updated hidden states
        """
        # 0. Establish residual connection
        residual = hidden_states

        # 1. Compute norm for hidden_states using pre-computed modulations
        norm_hidden_states, gate = self.norm(
            hidden_states=hidden_states,
            modulations=modulations,
        )

        # 2. Compute attention
        attn_output = self.attn(
            hidden_states=norm_hidden_states,
            image_rotary_emb=rotary_embeddings,
        )

        # 3. Apply feed-forward and projection
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
        """Apply MLP feed-forward and output projection."""
        feed_forward = self.proj_mlp(norm_hidden_states)
        mlp_hidden_states = nn.gelu_approx(feed_forward)
        hidden_states = mx.concatenate([attn_output, mlp_hidden_states], axis=2)
        gate = mx.expand_dims(gate, axis=1)
        hidden_states = gate * self.proj_out(hidden_states)
        return hidden_states
