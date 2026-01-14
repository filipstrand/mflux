"""Global modulation layers for FLUX.2.

FLUX.2 uses global modulation layers instead of per-block AdaLayerNorm:
- double_stream_modulation_img: Modulation for image stream in joint blocks
- double_stream_modulation_txt: Modulation for text stream in joint blocks
- single_stream_modulation: Modulation for single blocks

These are SINGLE global layers (not per-block), shared across all blocks.
The same modulation parameters are used for every block of that type.

Weight shapes from FLUX.2:
- double_stream_modulation_img.linear.weight: [36864, 6144] = [6 * 6144, 6144]
- double_stream_modulation_txt.linear.weight: [36864, 6144] = [6 * 6144, 6144]
- single_stream_modulation.linear.weight: [18432, 6144] = [3 * 6144, 6144]
"""

import mlx.core as mx
from mlx import nn


class DoubleStreamModulation(nn.Module):
    """Global modulation for double-stream (joint) blocks.

    Produces 6 modulation parameters (shared across all joint blocks):
    - shift_msa, scale_msa, gate_msa (for attention)
    - shift_mlp, scale_mlp, gate_mlp (for FFN)

    Args:
        hidden_dim: Hidden dimension (6144)
    """

    def __init__(self, hidden_dim: int = 6144):
        super().__init__()
        # Input: hidden_dim (6144 from time_guidance_embed)
        # Output: 6 * hidden_dim (36864)
        self.linear = nn.Linear(hidden_dim, 6 * hidden_dim, bias=False)
        self.hidden_dim = hidden_dim

    def __call__(self, conditioning: mx.array) -> tuple[mx.array, ...]:
        """Compute modulation parameters.

        Args:
            conditioning: Time/guidance embeddings [batch, hidden_dim]

        Returns:
            Tuple of (shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp)
            Each has shape [batch, hidden_dim]
        """
        modulation = self.linear(nn.silu(conditioning))
        chunk_size = self.hidden_dim

        shift_msa = modulation[:, 0 * chunk_size : 1 * chunk_size]
        scale_msa = modulation[:, 1 * chunk_size : 2 * chunk_size]
        gate_msa = modulation[:, 2 * chunk_size : 3 * chunk_size]
        shift_mlp = modulation[:, 3 * chunk_size : 4 * chunk_size]
        scale_mlp = modulation[:, 4 * chunk_size : 5 * chunk_size]
        gate_mlp = modulation[:, 5 * chunk_size : 6 * chunk_size]

        return shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp


class SingleStreamModulation(nn.Module):
    """Global modulation for single-stream blocks.

    Produces 3 modulation parameters (shared across all single blocks):
    - shift, scale, gate (combined for attention and MLP)

    Args:
        hidden_dim: Hidden dimension (6144)
    """

    def __init__(self, hidden_dim: int = 6144):
        super().__init__()
        # Input: hidden_dim (6144 from time_guidance_embed)
        # Output: 3 * hidden_dim (18432)
        self.linear = nn.Linear(hidden_dim, 3 * hidden_dim, bias=False)
        self.hidden_dim = hidden_dim

    def __call__(self, conditioning: mx.array) -> tuple[mx.array, mx.array, mx.array]:
        """Compute modulation parameters.

        Args:
            conditioning: Time/guidance embeddings [batch, hidden_dim]

        Returns:
            Tuple of (shift, scale, gate)
            Each has shape [batch, hidden_dim]
        """
        modulation = self.linear(nn.silu(conditioning))
        chunk_size = self.hidden_dim

        shift = modulation[:, 0 * chunk_size : 1 * chunk_size]
        scale = modulation[:, 1 * chunk_size : 2 * chunk_size]
        gate = modulation[:, 2 * chunk_size : 3 * chunk_size]

        return shift, scale, gate
