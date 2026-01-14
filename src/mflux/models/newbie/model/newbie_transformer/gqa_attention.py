"""Grouped Query Attention (GQA) for NewBie-image NextDiT.

GQA reduces memory and compute by sharing KV heads across multiple query heads.
NewBie uses 24 query heads with 8 KV heads (3:1 ratio).
"""

import mlx.core as mx
import mlx.nn as nn


class GQAttention(nn.Module):
    """Grouped Query Attention with RoPE.

    In GQA, multiple query heads share the same key-value heads.
    For NewBie: 24 query heads, 8 KV heads (3 query heads per KV head).

    Args:
        hidden_dim: Hidden dimension (2560 for NewBie)
        num_query_heads: Number of query heads (24 for NewBie)
        num_kv_heads: Number of key-value heads (8 for NewBie)
        head_dim: Dimension per head (64 for NewBie, derived from hidden_dim/num_query_heads)
        qk_norm: Whether to apply RMSNorm to Q and K
    """

    def __init__(
        self,
        hidden_dim: int = 2560,
        num_query_heads: int = 24,
        num_kv_heads: int = 8,
        head_dim: int | None = None,
        qk_norm: bool = True,
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_query_heads = num_query_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim if head_dim is not None else hidden_dim // num_query_heads
        self.num_groups = num_query_heads // num_kv_heads  # 3 for NewBie

        # Query projection - full heads
        self.wq = nn.Linear(hidden_dim, num_query_heads * self.head_dim, bias=False)

        # Key and Value projections - shared KV heads
        self.wk = nn.Linear(hidden_dim, num_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(hidden_dim, num_kv_heads * self.head_dim, bias=False)

        # Output projection
        self.wo = nn.Linear(num_query_heads * self.head_dim, hidden_dim, bias=False)

        # Optional QK normalization (RMSNorm)
        self.qk_norm = qk_norm
        if qk_norm:
            self.q_norm = nn.RMSNorm(self.head_dim, eps=1e-6)
            self.k_norm = nn.RMSNorm(self.head_dim, eps=1e-6)

        self.scale = self.head_dim ** -0.5

    def __call__(
        self,
        hidden_states: mx.array,
        rope_cos: mx.array | None = None,
        rope_sin: mx.array | None = None,
        attention_mask: mx.array | None = None,
    ) -> mx.array:
        """
        Forward pass for GQA.

        Args:
            hidden_states: Input tensor [batch, seq_len, hidden_dim]
            rope_cos: Cosine component for RoPE [seq_len, head_dim]
            rope_sin: Sine component for RoPE [seq_len, head_dim]
            attention_mask: Optional attention mask

        Returns:
            Output tensor [batch, seq_len, hidden_dim]
        """
        batch_size, seq_len, _ = hidden_states.shape

        # Project to Q, K, V
        q = self.wq(hidden_states)  # [batch, seq, num_query_heads * head_dim]
        k = self.wk(hidden_states)  # [batch, seq, num_kv_heads * head_dim]
        v = self.wv(hidden_states)  # [batch, seq, num_kv_heads * head_dim]

        # Reshape for multi-head attention
        q = q.reshape(batch_size, seq_len, self.num_query_heads, self.head_dim)
        k = k.reshape(batch_size, seq_len, self.num_kv_heads, self.head_dim)
        v = v.reshape(batch_size, seq_len, self.num_kv_heads, self.head_dim)

        # Apply QK normalization if enabled
        if self.qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)

        # Apply RoPE if provided
        if rope_cos is not None and rope_sin is not None:
            q = self._apply_rope(q, rope_cos, rope_sin)
            k = self._apply_rope(k, rope_cos, rope_sin)

        # Transpose for attention: [batch, heads, seq, head_dim]
        q = q.transpose(0, 2, 1, 3)
        k = k.transpose(0, 2, 1, 3)
        v = v.transpose(0, 2, 1, 3)

        # Expand KV heads to match query heads for GQA
        # Each KV head is repeated num_groups times
        k = mx.repeat(k, self.num_groups, axis=1)  # [batch, num_query_heads, seq, head_dim]
        v = mx.repeat(v, self.num_groups, axis=1)

        # Compute attention scores
        attn_weights = mx.matmul(q, k.transpose(0, 1, 3, 2)) * self.scale

        # Apply attention mask if provided
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        # Softmax
        attn_weights = mx.softmax(attn_weights, axis=-1)

        # Apply attention to values
        attn_output = mx.matmul(attn_weights, v)  # [batch, heads, seq, head_dim]

        # Reshape back: [batch, seq, heads, head_dim] -> [batch, seq, hidden]
        attn_output = attn_output.transpose(0, 2, 1, 3)
        attn_output = attn_output.reshape(batch_size, seq_len, -1)

        # Output projection
        output = self.wo(attn_output)

        return output

    def _apply_rope(
        self,
        x: mx.array,
        cos: mx.array,
        sin: mx.array,
    ) -> mx.array:
        """Apply rotary position embeddings.

        Args:
            x: Input tensor [batch, seq, heads, head_dim]
            cos: Cosine tensor [seq, head_dim] or [1, seq, 1, head_dim]
            sin: Sine tensor [seq, head_dim] or [1, seq, 1, head_dim]

        Returns:
            Tensor with RoPE applied
        """
        # Ensure cos/sin have right shape
        if cos.ndim == 2:
            cos = cos[None, :, None, :]  # [1, seq, 1, head_dim]
            sin = sin[None, :, None, :]

        # Split into two halves for rotation
        x1 = x[..., : self.head_dim // 2]
        x2 = x[..., self.head_dim // 2 :]

        cos = cos[..., : self.head_dim // 2]
        sin = sin[..., : self.head_dim // 2]

        # Apply rotation
        rotated = mx.concatenate([
            x1 * cos - x2 * sin,
            x1 * sin + x2 * cos,
        ], axis=-1)

        return rotated


class GQCrossAttention(nn.Module):
    """Grouped Query Cross-Attention for text conditioning.

    Cross-attention where queries come from image features and
    keys/values come from text embeddings.

    Args:
        hidden_dim: Image hidden dimension (2560)
        text_dim: Text embedding dimension (varies by encoder)
        num_query_heads: Number of query heads (24)
        num_kv_heads: Number of key-value heads (8)
        head_dim: Dimension per head (64)
    """

    def __init__(
        self,
        hidden_dim: int = 2560,
        text_dim: int = 2560,
        num_query_heads: int = 24,
        num_kv_heads: int = 8,
        head_dim: int | None = None,
        qk_norm: bool = True,
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.text_dim = text_dim
        self.num_query_heads = num_query_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim if head_dim is not None else hidden_dim // num_query_heads
        self.num_groups = num_query_heads // num_kv_heads

        # Query from image features
        self.wq = nn.Linear(hidden_dim, num_query_heads * self.head_dim, bias=False)

        # Key and Value from text features
        self.wk = nn.Linear(text_dim, num_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(text_dim, num_kv_heads * self.head_dim, bias=False)

        # Output projection
        self.wo = nn.Linear(num_query_heads * self.head_dim, hidden_dim, bias=False)

        # Optional QK normalization
        self.qk_norm = qk_norm
        if qk_norm:
            self.q_norm = nn.RMSNorm(self.head_dim, eps=1e-6)
            self.k_norm = nn.RMSNorm(self.head_dim, eps=1e-6)

        self.scale = self.head_dim ** -0.5

    def __call__(
        self,
        hidden_states: mx.array,
        encoder_hidden_states: mx.array,
        attention_mask: mx.array | None = None,
    ) -> mx.array:
        """
        Forward pass for cross-attention.

        Args:
            hidden_states: Image features [batch, img_seq, hidden_dim]
            encoder_hidden_states: Text features [batch, text_seq, text_dim]
            attention_mask: Optional attention mask

        Returns:
            Output tensor [batch, img_seq, hidden_dim]
        """
        batch_size, img_seq_len, _ = hidden_states.shape
        _, text_seq_len, _ = encoder_hidden_states.shape

        # Project queries from image, keys/values from text
        q = self.wq(hidden_states)
        k = self.wk(encoder_hidden_states)
        v = self.wv(encoder_hidden_states)

        # Reshape for multi-head attention
        q = q.reshape(batch_size, img_seq_len, self.num_query_heads, self.head_dim)
        k = k.reshape(batch_size, text_seq_len, self.num_kv_heads, self.head_dim)
        v = v.reshape(batch_size, text_seq_len, self.num_kv_heads, self.head_dim)

        # Apply QK normalization if enabled
        if self.qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)

        # Transpose for attention: [batch, heads, seq, head_dim]
        q = q.transpose(0, 2, 1, 3)
        k = k.transpose(0, 2, 1, 3)
        v = v.transpose(0, 2, 1, 3)

        # Expand KV heads for GQA
        k = mx.repeat(k, self.num_groups, axis=1)
        v = mx.repeat(v, self.num_groups, axis=1)

        # Compute attention
        attn_weights = mx.matmul(q, k.transpose(0, 1, 3, 2)) * self.scale

        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        attn_weights = mx.softmax(attn_weights, axis=-1)
        attn_output = mx.matmul(attn_weights, v)

        # Reshape back
        attn_output = attn_output.transpose(0, 2, 1, 3)
        attn_output = attn_output.reshape(batch_size, img_seq_len, -1)

        # Output projection
        output = self.wo(attn_output)

        return output
