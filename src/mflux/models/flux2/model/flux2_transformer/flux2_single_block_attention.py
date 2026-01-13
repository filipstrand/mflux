"""Fused single block attention for FLUX.2.

FLUX.2 single blocks use a fused QKV+MLP projection:
- to_qkv_mlp_proj: Projects input to Q, K, V, and MLP hidden states in one operation
- to_out: Projects concatenated attention output and MLP output back to hidden dim

This is more efficient than separate projections used in FLUX.1.
"""

import mlx.core as mx
from mlx import nn

from mflux.models.flux2.model.flux2_transformer.common.attention_utils import AttentionUtils


class Flux2SingleBlockAttention(nn.Module):
    """Fused attention for FLUX.2 single transformer blocks.

    Uses a single projection to compute Q, K, V, and MLP hidden states,
    then a single output projection after concatenating attention and MLP outputs.
    """

    def __init__(self):
        super().__init__()
        self.head_dimension = 128
        self.batch_size = 1
        self.num_heads = 48
        self.hidden_dim = 6144  # 48 * 128
        self.mlp_hidden_dim = 18432  # 6144 * 3 (mlp_ratio = 3.0)

        # Fused projection: hidden_dim -> Q + K + V + MLP_hidden
        # Output size: 6144 + 6144 + 6144 + 18432 = 36864
        self.to_qkv_mlp_proj = nn.Linear(
            self.hidden_dim,
            self.hidden_dim * 3 + self.mlp_hidden_dim,
            bias=False
        )

        # Output projection: attention_out + mlp_out -> hidden_dim
        # Input size: 6144 + 18432 = 24576
        self.to_out = nn.Linear(
            self.hidden_dim + self.mlp_hidden_dim,
            self.hidden_dim,
            bias=False
        )

        # Normalization layers for Q and K
        self.norm_q = nn.RMSNorm(self.head_dimension)
        self.norm_k = nn.RMSNorm(self.head_dimension)

    def __call__(
        self,
        hidden_states: mx.array,
        image_rotary_emb: mx.array,
    ) -> tuple[mx.array, mx.array]:
        """Compute fused attention and MLP.

        Args:
            hidden_states: Input tensor [batch, seq_len, hidden_dim]
            image_rotary_emb: Rotary embeddings

        Returns:
            Tuple of (attention_output, mlp_hidden_states)
            These will be concatenated and projected in the single block.
        """
        batch_size = hidden_states.shape[0]
        seq_len = hidden_states.shape[1]

        # 1. Compute fused QKV + MLP projection
        qkv_mlp = self.to_qkv_mlp_proj(hidden_states)

        # 2. Split into Q, K, V, and MLP hidden
        query = qkv_mlp[:, :, : self.hidden_dim]
        key = qkv_mlp[:, :, self.hidden_dim : 2 * self.hidden_dim]
        value = qkv_mlp[:, :, 2 * self.hidden_dim : 3 * self.hidden_dim]
        mlp_hidden = qkv_mlp[:, :, 3 * self.hidden_dim :]

        # 3. Reshape Q, K, V for multi-head attention [B, S, H*D] -> [B, H, S, D]
        query = mx.transpose(
            mx.reshape(query, (batch_size, seq_len, self.num_heads, self.head_dimension)),
            (0, 2, 1, 3)
        )
        key = mx.transpose(
            mx.reshape(key, (batch_size, seq_len, self.num_heads, self.head_dimension)),
            (0, 2, 1, 3)
        )
        value = mx.transpose(
            mx.reshape(value, (batch_size, seq_len, self.num_heads, self.head_dimension)),
            (0, 2, 1, 3)
        )

        # 4. Apply normalization to Q and K
        q_dtype = query.dtype
        k_dtype = key.dtype
        query = self.norm_q(query.astype(mx.float32)).astype(q_dtype)
        key = self.norm_k(key.astype(mx.float32)).astype(k_dtype)

        # 5. Apply RoPE
        query, key = AttentionUtils.apply_rope(xq=query, xk=key, freqs_cis=image_rotary_emb)

        # 6. Compute attention
        attn_output = AttentionUtils.compute_attention(
            query=query,
            key=key,
            value=value,
            batch_size=batch_size,
            num_heads=self.num_heads,
            head_dim=self.head_dimension,
        )

        return attn_output, mlp_hidden

    def project_output(
        self,
        attn_output: mx.array,
        mlp_hidden: mx.array,
    ) -> mx.array:
        """Project concatenated attention and MLP outputs.

        Args:
            attn_output: Attention output [batch, seq_len, hidden_dim]
            mlp_hidden: MLP hidden states [batch, seq_len, mlp_hidden_dim]

        Returns:
            Projected output [batch, seq_len, hidden_dim]
        """
        # Apply GELU to MLP hidden states
        mlp_output = nn.gelu_approx(mlp_hidden)

        # Concatenate attention and MLP outputs
        combined = mx.concatenate([attn_output, mlp_output], axis=2)

        # Project back to hidden_dim
        return self.to_out(combined)
