"""
Chroma Adaptive Layer Normalization modules.

These differ from FLUX's AdaLayerNorm in that they don't have a linear projection layer.
The modulations are pre-computed by the DistilledGuidanceLayer and passed directly.
"""

import mlx.core as mx
from mlx import nn


class ChromaAdaLayerNormZeroPruned(nn.Module):
    """
    Adaptive LayerNorm for Chroma joint transformer blocks.

    Takes 6 pre-computed modulation vectors and applies them:
    - shift_msa, scale_msa, gate_msa for attention
    - shift_mlp, scale_mlp, gate_mlp for feed-forward

    Unlike FLUX's AdaLayerNormZero, this has NO linear layer.
    The modulations come directly from DistilledGuidanceLayer.
    """

    def __init__(self, dim: int = 3072, eps: float = 1e-6):
        super().__init__()
        self.dim = dim
        self.norm = nn.LayerNorm(dims=dim, eps=eps, affine=False)

    def __call__(
        self,
        hidden_states: mx.array,
        modulations: mx.array,
    ) -> tuple[mx.array, mx.array, mx.array, mx.array, mx.array]:
        """
        Apply adaptive layer normalization.

        Args:
            hidden_states: Input tensor [batch, seq_len, dim]
            modulations: Pre-computed modulations [batch, 6, dim]

        Returns:
            Tuple of (normalized_hidden_states, gate_msa, shift_mlp, scale_mlp, gate_mlp)
        """
        # Flatten [batch, 6, dim] to [batch, 6*dim] then chunk
        modulations_flat = mx.reshape(modulations, (modulations.shape[0], -1))
        chunk_size = self.dim

        shift_msa = modulations_flat[:, 0 * chunk_size : 1 * chunk_size]
        scale_msa = modulations_flat[:, 1 * chunk_size : 2 * chunk_size]
        gate_msa = modulations_flat[:, 2 * chunk_size : 3 * chunk_size]
        shift_mlp = modulations_flat[:, 3 * chunk_size : 4 * chunk_size]
        scale_mlp = modulations_flat[:, 4 * chunk_size : 5 * chunk_size]
        gate_mlp = modulations_flat[:, 5 * chunk_size : 6 * chunk_size]

        # Apply adaptive normalization
        hidden_states = self.norm(hidden_states) * (1 + scale_msa[:, None]) + shift_msa[:, None]

        return hidden_states, gate_msa, shift_mlp, scale_mlp, gate_mlp


class ChromaAdaLayerNormZeroSinglePruned(nn.Module):
    """
    Adaptive LayerNorm for Chroma single transformer blocks.

    Takes 3 pre-computed modulation vectors and applies them:
    - shift_msa, scale_msa, gate_msa for attention

    Unlike FLUX's AdaLayerNormZeroSingle, this has NO linear layer.
    The modulations come directly from DistilledGuidanceLayer.
    """

    def __init__(self, dim: int = 3072, eps: float = 1e-6):
        super().__init__()
        self.dim = dim
        self.norm = nn.LayerNorm(dims=dim, eps=eps, affine=False)

    def __call__(
        self,
        hidden_states: mx.array,
        modulations: mx.array,
    ) -> tuple[mx.array, mx.array]:
        """
        Apply adaptive layer normalization.

        Args:
            hidden_states: Input tensor [batch, seq_len, dim]
            modulations: Pre-computed modulations [batch, 3, dim]

        Returns:
            Tuple of (normalized_hidden_states, gate_msa)
        """
        # Flatten [batch, 3, dim] to [batch, 3*dim] then chunk
        modulations_flat = mx.reshape(modulations, (modulations.shape[0], -1))
        chunk_size = self.dim

        shift_msa = modulations_flat[:, 0 * chunk_size : 1 * chunk_size]
        scale_msa = modulations_flat[:, 1 * chunk_size : 2 * chunk_size]
        gate_msa = modulations_flat[:, 2 * chunk_size : 3 * chunk_size]

        # Apply adaptive normalization
        hidden_states = self.norm(hidden_states) * (1 + scale_msa[:, None]) + shift_msa[:, None]

        return hidden_states, gate_msa


class ChromaAdaLayerNormContinuousPruned(nn.Module):
    """
    Adaptive LayerNorm for the final output in Chroma.

    Takes 2 pre-computed modulation vectors (shift and scale).

    Unlike FLUX's AdaLayerNormContinuous, this has NO linear layer.
    The modulations come directly from DistilledGuidanceLayer.
    """

    def __init__(self, dim: int = 3072, eps: float = 1e-6):
        super().__init__()
        self.dim = dim
        self.norm = nn.LayerNorm(dims=dim, eps=eps, affine=False)

    def __call__(
        self,
        hidden_states: mx.array,
        modulations: mx.array,
    ) -> mx.array:
        """
        Apply adaptive layer normalization.

        Args:
            hidden_states: Input tensor [batch, seq_len, dim]
            modulations: Pre-computed modulations [batch, 2, dim]

        Returns:
            Normalized hidden states
        """
        # Flatten [batch, 2, dim] to [batch, 2*dim] then chunk
        modulations_flat = mx.reshape(modulations, (modulations.shape[0], -1))
        chunk_size = self.dim

        shift = modulations_flat[:, 0 * chunk_size : 1 * chunk_size]
        scale = modulations_flat[:, 1 * chunk_size : 2 * chunk_size]

        # Apply adaptive normalization
        hidden_states = self.norm(hidden_states) * (1 + scale[:, None]) + shift[:, None]

        return hidden_states
