"""Hunyuan-DiT Attention Layers.

Hunyuan uses separate self-attention and cross-attention layers,
unlike FLUX which uses joint/multimodal attention.

Self-attention:
- Applied over image tokens
- Uses RoPE (Rotary Position Embeddings)
- QKV normalization with LayerNorm (per-head)

Cross-attention:
- Queries from image, keys/values from text encoder
- No RoPE (text positions not needed)
- Handles projected text embeddings
"""

import mlx.core as mx
from mlx import nn
from mlx.core.fast import scaled_dot_product_attention


class HunyuanQKNorm(nn.Module):
    """QK normalization layer for Hunyuan attention.

    A simple layer norm applied per-head to Q or K.
    """

    def __init__(self, head_dim: int):
        super().__init__()
        self.weight = mx.ones((head_dim,))
        self.bias = mx.zeros((head_dim,))
        self.eps = 1e-6

    def __call__(self, x: mx.array) -> mx.array:
        """Apply layer norm to the last dimension."""
        # x is [batch, heads, seq, head_dim]
        mean = mx.mean(x, axis=-1, keepdims=True)
        var = mx.var(x, axis=-1, keepdims=True)
        normalized = (x - mean) / mx.sqrt(var + self.eps)
        return normalized * self.weight + self.bias


class HunyuanSelfAttention(nn.Module):
    """Self-attention layer for Hunyuan-DiT.

    Attends over image tokens with RoPE position embeddings.

    Args:
        hidden_dim: Hidden dimension (1408 for Hunyuan-DiT)
        num_heads: Number of attention heads (16 for Hunyuan-DiT)
        head_dim: Dimension per head (88 for Hunyuan-DiT)
    """

    def __init__(
        self,
        hidden_dim: int = 1408,
        num_heads: int = 16,
        head_dim: int = 88,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = head_dim

        # QKV projections
        self.to_q = nn.Linear(hidden_dim, num_heads * head_dim, bias=True)
        self.to_k = nn.Linear(hidden_dim, num_heads * head_dim, bias=True)
        self.to_v = nn.Linear(hidden_dim, num_heads * head_dim, bias=True)

        # Output projection
        self.to_out = nn.Linear(num_heads * head_dim, hidden_dim, bias=True)

        # QK normalization (per-head LayerNorm with bias)
        self.norm_q = HunyuanQKNorm(head_dim)
        self.norm_k = HunyuanQKNorm(head_dim)

    def __call__(
        self,
        hidden_states: mx.array,
        rotary_emb: mx.array | None = None,
    ) -> mx.array:
        """
        Forward pass of self-attention.

        Args:
            hidden_states: Input tensor [batch, seq_len, hidden_dim]
            rotary_emb: Rotary position embeddings [seq_len, head_dim, 2]

        Returns:
            Output tensor [batch, seq_len, hidden_dim]
        """
        batch_size, seq_len, _ = hidden_states.shape

        # Compute Q, K, V
        query = self.to_q(hidden_states)
        key = self.to_k(hidden_states)
        value = self.to_v(hidden_states)

        # Reshape to [batch, num_heads, seq_len, head_dim]
        query = query.reshape(batch_size, seq_len, self.num_heads, self.head_dim)
        query = query.transpose(0, 2, 1, 3)

        key = key.reshape(batch_size, seq_len, self.num_heads, self.head_dim)
        key = key.transpose(0, 2, 1, 3)

        value = value.reshape(batch_size, seq_len, self.num_heads, self.head_dim)
        value = value.transpose(0, 2, 1, 3)

        # Apply QK normalization in float32 for stability
        q_dtype = query.dtype
        k_dtype = key.dtype
        query = self.norm_q(query.astype(mx.float32)).astype(q_dtype)
        key = self.norm_k(key.astype(mx.float32)).astype(k_dtype)

        # Apply rotary embeddings if provided
        if rotary_emb is not None:
            query, key = self._apply_rope(query, key, rotary_emb)

        # Scaled dot-product attention
        scale = 1.0 / mx.sqrt(mx.array(self.head_dim, dtype=mx.float32))
        hidden_states = scaled_dot_product_attention(query, key, value, scale=scale)

        # Reshape back to [batch, seq_len, hidden_dim]
        hidden_states = hidden_states.transpose(0, 2, 1, 3)
        hidden_states = hidden_states.reshape(batch_size, seq_len, self.num_heads * self.head_dim)

        # Output projection
        hidden_states = self.to_out(hidden_states)

        return hidden_states

    @staticmethod
    def _apply_rope(
        query: mx.array,
        key: mx.array,
        rotary_emb: mx.array,
    ) -> tuple[mx.array, mx.array]:
        """Apply rotary position embeddings to query and key.

        Args:
            query: Query tensor [batch, num_heads, seq_len, head_dim]
            key: Key tensor [batch, num_heads, seq_len, head_dim]
            rotary_emb: Rotary embeddings [seq_len, head_dim//2, 2] or [1, seq_len, 1, head_dim//2, 2]

        Returns:
            Tuple of (rotated_query, rotated_key)
        """
        query_f = query.astype(mx.float32)
        key_f = key.astype(mx.float32)

        # Reshape for rotation
        query_r = query_f.reshape(*query_f.shape[:-1], -1, 2)
        key_r = key_f.reshape(*key_f.shape[:-1], -1, 2)

        # Get cos and sin from rotary_emb
        cos = rotary_emb[..., 0]
        sin = rotary_emb[..., 1]

        # Ensure broadcasting works [batch, heads, seq, pairs, 2]
        if cos.ndim == 2:
            cos = cos.reshape(1, 1, cos.shape[0], cos.shape[1])
            sin = sin.reshape(1, 1, sin.shape[0], sin.shape[1])

        # Apply rotation: (x * cos - y * sin, x * sin + y * cos)
        query_out_r = query_r[..., 0] * cos - query_r[..., 1] * sin
        query_out_i = query_r[..., 0] * sin + query_r[..., 1] * cos
        query_out = mx.stack([query_out_r, query_out_i], axis=-1)
        query_out = query_out.reshape(*query_f.shape)

        key_out_r = key_r[..., 0] * cos - key_r[..., 1] * sin
        key_out_i = key_r[..., 0] * sin + key_r[..., 1] * cos
        key_out = mx.stack([key_out_r, key_out_i], axis=-1)
        key_out = key_out.reshape(*key_f.shape)

        return query_out.astype(query.dtype), key_out.astype(key.dtype)


class HunyuanCrossAttention(nn.Module):
    """Cross-attention layer for Hunyuan-DiT.

    Queries from image tokens, keys/values from text encoder output.

    Args:
        hidden_dim: Hidden dimension for image (1408 for Hunyuan-DiT)
        text_dim: Text encoder hidden dimension (projected to hidden_dim)
        num_heads: Number of attention heads (16 for Hunyuan-DiT)
        head_dim: Dimension per head (88 for Hunyuan-DiT)
    """

    def __init__(
        self,
        hidden_dim: int = 1408,
        text_dim: int = 1408,  # After projection, text is also hidden_dim
        num_heads: int = 16,
        head_dim: int = 88,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.text_dim = text_dim
        self.num_heads = num_heads
        self.head_dim = head_dim

        # Query projection (from image)
        self.to_q = nn.Linear(hidden_dim, num_heads * head_dim, bias=True)

        # Key, Value projections (from text - uses text_dim which is 1024 after projection)
        # Based on HF weights: to_k has shape (1408, 1024), to_v has shape (1408, 1024)
        self.to_k = nn.Linear(text_dim, num_heads * head_dim, bias=True)
        self.to_v = nn.Linear(text_dim, num_heads * head_dim, bias=True)

        # Output projection
        self.to_out = nn.Linear(num_heads * head_dim, hidden_dim, bias=True)

        # QK normalization (per-head LayerNorm with bias)
        self.norm_q = HunyuanQKNorm(head_dim)
        self.norm_k = HunyuanQKNorm(head_dim)

    def __call__(
        self,
        hidden_states: mx.array,
        encoder_hidden_states: mx.array,
        attention_mask: mx.array | None = None,
    ) -> mx.array:
        """
        Forward pass of cross-attention.

        Args:
            hidden_states: Image hidden states [batch, img_seq_len, hidden_dim]
            encoder_hidden_states: Text encoder output [batch, text_seq_len, text_dim]
            attention_mask: Optional attention mask for text padding

        Returns:
            Output tensor [batch, img_seq_len, hidden_dim]
        """
        batch_size, img_seq_len, _ = hidden_states.shape
        text_seq_len = encoder_hidden_states.shape[1]

        # Compute Q from image
        query = self.to_q(hidden_states)

        # Compute K, V from text
        key = self.to_k(encoder_hidden_states)
        value = self.to_v(encoder_hidden_states)

        # Reshape to [batch, num_heads, seq_len, head_dim]
        query = query.reshape(batch_size, img_seq_len, self.num_heads, self.head_dim)
        query = query.transpose(0, 2, 1, 3)

        key = key.reshape(batch_size, text_seq_len, self.num_heads, self.head_dim)
        key = key.transpose(0, 2, 1, 3)

        value = value.reshape(batch_size, text_seq_len, self.num_heads, self.head_dim)
        value = value.transpose(0, 2, 1, 3)

        # Apply QK normalization in float32 for stability
        q_dtype = query.dtype
        k_dtype = key.dtype
        query = self.norm_q(query.astype(mx.float32)).astype(q_dtype)
        key = self.norm_k(key.astype(mx.float32)).astype(k_dtype)

        # Convert attention mask to additive mask if provided
        if attention_mask is not None:
            attention_mask = self._convert_mask_to_additive(attention_mask)

        # Scaled dot-product attention
        scale = 1.0 / mx.sqrt(mx.array(self.head_dim, dtype=mx.float32))
        hidden_states = scaled_dot_product_attention(
            query, key, value, scale=scale, mask=attention_mask
        )

        # Reshape back to [batch, seq_len, hidden_dim]
        hidden_states = hidden_states.transpose(0, 2, 1, 3)
        hidden_states = hidden_states.reshape(batch_size, img_seq_len, self.num_heads * self.head_dim)

        # Output projection
        hidden_states = self.to_out(hidden_states)

        return hidden_states

    @staticmethod
    def _convert_mask_to_additive(mask: mx.array) -> mx.array:
        """Convert boolean/float mask to additive attention mask.

        Args:
            mask: Input mask [batch, seq_len] where 1 = valid, 0 = masked

        Returns:
            Additive mask [batch, 1, 1, seq_len] with -inf for masked positions
        """
        additive = (1.0 - mask.astype(mx.float32)) * (-1e9)
        return additive.reshape(mask.shape[0], 1, 1, mask.shape[1])
