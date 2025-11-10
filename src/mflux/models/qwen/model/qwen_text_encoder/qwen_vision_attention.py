import mlx.core as mx
from mlx import nn


class VisionAttention(nn.Module):
    def __init__(self, embed_dim: int = 1280, num_heads: int = 16):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.qkv = nn.Linear(embed_dim, 3 * embed_dim, bias=True)
        self.proj = nn.Linear(embed_dim, embed_dim, bias=True)

    def __call__(self, x: mx.array, position_embeddings=None, cu_seqlens=None) -> mx.array:
        # For vision attention, x is [seq_len, embed_dim] (no batch dimension)
        seq_len, embed_dim = x.shape

        # Compute QKV
        qkv = self.qkv(x)  # [seq_len, 3 * embed_dim]
        qkv = qkv.reshape(seq_len, 3, self.num_heads, self.head_dim)
        q, k, v = mx.split(qkv, 3, axis=1)
        q = q.squeeze(1).transpose(1, 0, 2)  # [heads, seq, head_dim]
        k = k.squeeze(1).transpose(1, 0, 2)
        v = v.squeeze(1).transpose(1, 0, 2)

        # Apply rotary position embeddings if provided
        if position_embeddings is not None:
            cos_emb, sin_emb = position_embeddings
            # Ensure embeddings match sequence length
            if cos_emb.shape[0] != seq_len:
                cos_emb = cos_emb[:seq_len]
                sin_emb = sin_emb[:seq_len]

            # Apply RoPE to q and k (matching HF's rotate_half logic)
            # Matching QwenAttention._apply_multimodal_rotary_pos_emb implementation
            def apply_rope(x, cos, sin):
                # x: [heads, seq, head_dim], cos/sin: [seq, head_dim]
                # Expand cos/sin to match head dimension for broadcasting
                cos_expanded = mx.expand_dims(cos, axis=0)  # [1, seq, head_dim]
                sin_expanded = mx.expand_dims(sin, axis=0)  # [1, seq, head_dim]

                # Standard RoPE formula: (q * cos) + (rotate_half(q) * sin)
                # where rotate_half([x1, x2]) = [-x2, x1]
                def rotate_half(x):
                    x1 = x[..., : x.shape[-1] // 2]
                    x2 = x[..., x.shape[-1] // 2 :]
                    return mx.concatenate([-x2, x1], axis=-1)

                rotated = (x * cos_expanded) + (rotate_half(x) * sin_expanded)
                return rotated

            q = apply_rope(q, cos_emb, sin_emb)
            k = apply_rope(k, cos_emb, sin_emb)

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

            # Process each chunk separately
            attn_outputs = []
            scale = 1.0 / (self.head_dim**0.5)
            for q_chunk, k_chunk, v_chunk in zip(q_chunks, k_chunks, v_chunks):
                # Compute attention for this chunk
                scores = mx.matmul(q_chunk, k_chunk.transpose(0, 2, 1)) * scale
                attn_weights = mx.softmax(scores, axis=-1)
                attn_chunk = mx.matmul(attn_weights, v_chunk)  # [heads, chunk_len, head_dim]
                attn_outputs.append(attn_chunk)

            # Concatenate chunks back together
            attn_output = mx.concatenate(attn_outputs, axis=1)  # [heads, seq, head_dim]
        else:
            # Full attention (no chunking)
            scale = 1.0 / (self.head_dim**0.5)
            scores = mx.matmul(q, k.transpose(0, 2, 1)) * scale  # [heads, seq, seq]
            attn_weights = mx.softmax(scores, axis=-1)
            attn_output = mx.matmul(attn_weights, v)  # [heads, seq, head_dim]

        # Reshape and project
        attn_output = attn_output.transpose(1, 0, 2).reshape(seq_len, embed_dim)  # [seq, embed_dim]
        return self.proj(attn_output)
