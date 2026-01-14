"""Single transformer block for FLUX.2.

FLUX.2 single blocks are fundamentally different from FLUX.1:
- Fused QKV+MLP projection (to_qkv_mlp_proj) for efficiency
- Global modulation from SingleStreamModulation layers
- Combined output projection after concatenating attention and MLP outputs
"""

import mlx.core as mx
from mlx import nn

from mflux.models.flux2.model.flux2_transformer.flux2_single_block_attention import Flux2SingleBlockAttention


class Flux2SingleTransformerBlock(nn.Module):
    """Single transformer block for FLUX.2.

    Uses fused QKV+MLP projection for efficiency.
    Modulation parameters are provided externally from global modulation layers.
    """

    def __init__(self, layer: int):
        super().__init__()
        self.layer = layer
        self.hidden_dim = 6144

        # Layer norm (applied before attention, no affine - modulation comes externally)
        self.norm = nn.LayerNorm(dims=self.hidden_dim, eps=1e-6, affine=False)

        # Fused attention with QKV+MLP projection
        self.attn = Flux2SingleBlockAttention()

    def __call__(
        self,
        hidden_states: mx.array,
        rotary_embeddings: mx.array,
        modulation: tuple[mx.array, mx.array, mx.array],
    ) -> mx.array:
        """Process single transformer block.

        Args:
            hidden_states: Input tensor [batch, seq_len, hidden_dim]
            rotary_embeddings: RoPE embeddings
            modulation: (shift, scale, gate) from SingleStreamModulation

        Returns:
            Output tensor [batch, seq_len, hidden_dim]
        """
        # Unpack modulation parameters
        shift, scale, gate = modulation

        # Store residual
        residual = hidden_states

        # 1. Apply layer norm and modulation
        norm_hidden_states = self.norm(hidden_states)
        norm_hidden_states = norm_hidden_states * (1 + scale[:, None]) + shift[:, None]

        # 2. Compute fused attention and MLP
        attn_output, mlp_hidden = self.attn(
            hidden_states=norm_hidden_states,
            image_rotary_emb=rotary_embeddings,
        )

        # 3. Project combined outputs
        output = self.attn.project_output(attn_output, mlp_hidden)

        # 4. Apply gating and residual
        output = mx.expand_dims(gate, axis=1) * output
        hidden_states = residual + output

        return hidden_states
