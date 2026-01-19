import mlx.core as mx
from mlx import nn
from mlx.core.fast import scaled_dot_product_attention


class VisionAttention(nn.Module):
    """Multi-head self-attention for vision encoder with RoPE support.

    This attention module is designed for processing unbatched visual sequences
    (shape: [seq_len, embed_dim]) and supports optional windowed attention via
    cumulative sequence lengths (cu_seqlens).

    NOTE: This module does NOT support attention masking. For controlling
    attention patterns, use cu_seqlens for windowed/chunked attention.
    This is by design - visual features typically attend to all positions
    within their window without masking.

    Unlike QwenAttention (text encoder), this operates on unbatched input:
    - VisionAttention: input [seq_len, embed_dim], Q/K/V [heads, seq, head_dim]
    - QwenAttention: input [batch, seq_len, hidden], Q/K/V [batch, heads, seq, head_dim]

    The batch dimension is added internally only for scaled_dot_product_attention.
    """

    def __init__(self, embed_dim: int = 1280, num_heads: int = 16):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.qkv = nn.Linear(embed_dim, 3 * embed_dim, bias=True)
        self.proj = nn.Linear(embed_dim, embed_dim, bias=True)

    def _rotate_half(self, x: mx.array) -> mx.array:
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return mx.concatenate([-x2, x1], axis=-1)

    def _apply_rope(self, x: mx.array, cos: mx.array, sin: mx.array) -> mx.array:
        cos_expanded = mx.expand_dims(cos, axis=0)
        sin_expanded = mx.expand_dims(sin, axis=0)
        rotated = (x * cos_expanded) + (self._rotate_half(x) * sin_expanded)
        return rotated

    def __call__(self, x: mx.array, position_embeddings=None, cu_seqlens=None) -> mx.array:
        """Compute vision attention.

        Args:
            x: Input tensor of shape [seq_len, embed_dim] (unbatched).
            position_embeddings: Optional tuple of (cos, sin) RoPE embeddings,
                each of shape [seq_len, head_dim].
            cu_seqlens: Optional cumulative sequence lengths for windowed attention.
                If provided with >2 elements, splits input into chunks where each
                chunk attends only within itself.

        Returns:
            Output tensor of shape [seq_len, embed_dim].
        """
        seq_len, embed_dim = x.shape

        qkv = self.qkv(x)
        qkv = qkv.reshape(seq_len, 3, self.num_heads, self.head_dim)
        q, k, v = mx.split(qkv, 3, axis=1)
        q = q.squeeze(1).transpose(1, 0, 2)
        k = k.squeeze(1).transpose(1, 0, 2)
        v = v.squeeze(1).transpose(1, 0, 2)

        if position_embeddings is not None:
            cos_emb, sin_emb = position_embeddings
            if cos_emb.shape[0] != seq_len:
                cos_emb = cos_emb[:seq_len]
                sin_emb = sin_emb[:seq_len]

            q = self._apply_rope(q, cos_emb, sin_emb)
            k = self._apply_rope(k, cos_emb, sin_emb)

        # Process attention chunks if cu_seqlens is provided (windowed attention)
        if cu_seqlens is not None and len(cu_seqlens) > 2:
            # Split Q, K, V into chunks based on cu_seqlens
            # cu_seqlens is cumulative, so lengths[i] = cu_seqlens[i+1] - cu_seqlens[i]
            lengths = [int((cu_seqlens[i + 1] - cu_seqlens[i]).item()) for i in range(len(cu_seqlens) - 1)]

            # Split tensors (q,k,v are [heads, seq, head_dim])
            q_chunks = []
            k_chunks = []
            v_chunks = []
            offset = 0
            for length in lengths:
                q_chunks.append(q[:, offset : offset + length, :])
                k_chunks.append(k[:, offset : offset + length, :])
                v_chunks.append(v[:, offset : offset + length, :])
                offset += length

            # Process each chunk separately using optimized Metal kernel
            attn_outputs = []
            scale = 1.0 / (self.head_dim**0.5)
            for q_chunk, k_chunk, v_chunk in zip(q_chunks, k_chunks, v_chunks):
                # Add batch dimension for scaled_dot_product_attention: [heads, seq, dim] -> [1, heads, seq, dim]
                q_batch = q_chunk[None, :, :, :]
                k_batch = k_chunk[None, :, :, :]
                v_batch = v_chunk[None, :, :, :]
                attn_chunk = scaled_dot_product_attention(q_batch, k_batch, v_batch, scale=scale)
                attn_outputs.append(attn_chunk[0])  # Remove batch dimension

            # Concatenate chunks back together
            attn_output = mx.concatenate(attn_outputs, axis=1)  # [heads, seq, head_dim]
        else:
            # Full attention (no chunking) - use optimized Metal kernel
            scale = 1.0 / (self.head_dim**0.5)
            # Add batch dimension: [heads, seq, dim] -> [1, heads, seq, dim]
            q_batch = q[None, :, :, :]
            k_batch = k[None, :, :, :]
            v_batch = v[None, :, :, :]
            attn_output = scaled_dot_product_attention(q_batch, k_batch, v_batch, scale=scale)
            attn_output = attn_output[0]  # Remove batch dimension: [heads, seq, head_dim]

        # Reshape and project
        attn_output = attn_output.transpose(1, 0, 2).reshape(seq_len, embed_dim)  # [seq, embed_dim]
        return self.proj(attn_output)
