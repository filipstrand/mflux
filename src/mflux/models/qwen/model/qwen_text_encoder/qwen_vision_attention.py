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

    def _rotate_half(self, x: mx.array) -> mx.array:
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return mx.concatenate([-x2, x1], axis=-1)

    def _apply_rope(self, x: mx.array, cos: mx.array, sin: mx.array) -> mx.array:
        orig_dtype = x.dtype
        x = x.astype(mx.float32)
        cos_expanded = mx.expand_dims(cos, axis=0).astype(mx.float32)
        sin_expanded = mx.expand_dims(sin, axis=0).astype(mx.float32)
        rotated = (x * cos_expanded) + (self._rotate_half(x) * sin_expanded)
        return rotated.astype(orig_dtype)

    def __call__(self, x: mx.array, position_embeddings=None, cu_seqlens=None) -> mx.array:
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

        scale = 1.0 / (self.head_dim**0.5)

        # Process attention chunks if cu_seqlens is provided (windowed attention)
        if cu_seqlens is not None and len(cu_seqlens) > 2:
            lengths = [int((cu_seqlens[i + 1] - cu_seqlens[i]).item()) for i in range(len(cu_seqlens) - 1)]

            attn_outputs = []
            offset = 0
            for length in lengths:
                q_chunk = mx.expand_dims(q[:, offset : offset + length, :], axis=0)
                k_chunk = mx.expand_dims(k[:, offset : offset + length, :], axis=0)
                v_chunk = mx.expand_dims(v[:, offset : offset + length, :], axis=0)
                offset += length
                out = mx.fast.scaled_dot_product_attention(q_chunk, k_chunk, v_chunk, scale=scale)
                attn_outputs.append(out.squeeze(0))

            attn_output = mx.concatenate(attn_outputs, axis=1)  # [heads, seq, head_dim]
        else:
            # Full attention (no chunking)
            q_4d = mx.expand_dims(q, axis=0)
            k_4d = mx.expand_dims(k, axis=0)
            v_4d = mx.expand_dims(v, axis=0)
            attn_output = mx.fast.scaled_dot_product_attention(q_4d, k_4d, v_4d, scale=scale)
            attn_output = attn_output.squeeze(0)  # [heads, seq, head_dim]

        # Reshape and project
        attn_output = attn_output.transpose(1, 0, 2).reshape(seq_len, embed_dim)  # [seq, embed_dim]
        return self.proj(attn_output)
