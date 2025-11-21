import math

import mlx.core as mx
from mlx import nn
from mlx.core.fast import scaled_dot_product_attention

from mflux.models.fibo_vlm.model.qwen3_vl_rms_norm import Qwen3VLRMSNorm


class Qwen3VLAttention(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        num_key_value_heads: int,
        head_dim: int,
        max_position_embeddings: int = 262144,
        rope_theta: float = 1000000.0,
        mrope_section: list[int] | None = None,
        attention_bias: bool = False,
        rms_norm_eps: float = 1e-6,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim
        self.num_key_value_groups = num_attention_heads // num_key_value_heads
        self.scaling = 1.0 / math.sqrt(self.head_dim)

        # Projections
        self.q_proj = nn.Linear(hidden_size, num_attention_heads * head_dim, bias=attention_bias)
        self.k_proj = nn.Linear(hidden_size, num_key_value_heads * head_dim, bias=attention_bias)
        self.v_proj = nn.Linear(hidden_size, num_key_value_heads * head_dim, bias=attention_bias)
        self.o_proj = nn.Linear(num_attention_heads * head_dim, hidden_size, bias=attention_bias)

        # Per-head RMSNorm for q/k (matches Qwen3VLTextAttention)
        self.q_norm = Qwen3VLRMSNorm(head_dim, eps=rms_norm_eps)
        self.k_norm = Qwen3VLRMSNorm(head_dim, eps=rms_norm_eps)

        # Multimodal RoPE config (values already baked into provided cos/sin)
        self.mrope_section = mrope_section or [24, 20, 20]

    def __call__(
        self,
        hidden_states: mx.array,
        attention_mask: mx.array | None = None,
        position_embeddings: tuple[mx.array, mx.array] | None = None,
        past_key_value: tuple[mx.array, mx.array] | None = None,
    ) -> mx.array | tuple[mx.array, tuple[mx.array, mx.array]]:
        bsz, q_len, _ = hidden_states.shape

        # Project to q, k, v
        q_proj = self.q_proj(hidden_states)
        k_proj = self.k_proj(hidden_states)
        v_proj = self.v_proj(hidden_states)

        # Reshape to (batch_size, seq_len, num_heads, head_dim)
        query_states = q_proj.reshape(bsz, q_len, self.num_attention_heads, self.head_dim)
        key_states = k_proj.reshape(bsz, q_len, self.num_key_value_heads, self.head_dim)
        value_states = v_proj.reshape(bsz, q_len, self.num_key_value_heads, self.head_dim)

        # Apply per-head RMSNorm before transposing (matches Qwen3VLTextAttention)
        query_states = self.q_norm(query_states)
        key_states = self.k_norm(key_states)

        # Transpose to (batch_size, num_heads, seq_len, head_dim)
        query_states = query_states.transpose(0, 2, 1, 3)
        key_states = key_states.transpose(0, 2, 1, 3)
        value_states = value_states.transpose(0, 2, 1, 3)

        # Apply rotary position embedding – cos/sin already include mRoPE interleaving
        if position_embeddings is not None:
            cos, sin = position_embeddings

            # Qwen3VLRotaryEmbedding now returns cos/sin with shape (batch, seq_len, head_dim)
            # after apply_interleaved_mrope, matching PyTorch's Qwen3VLTextRotaryEmbedding
            query_states, key_states = self._apply_rotary_pos_emb(
                q=query_states,
                k=key_states,
                cos=cos,
                sin=sin,
            )

        # Store key/value states before GQA repetition for caching
        # These are the original (batch, num_kv_heads, q_len, head_dim) tensors
        cache_key_states = key_states
        cache_value_states = value_states

        # KV cache: concatenate with past key/value states if provided
        if past_key_value is not None:
            past_key, past_value = past_key_value
            # past_key/past_value shape: (batch, num_kv_heads, cached_seq_len, head_dim)
            # cache_key_states/cache_value_states shape: (batch, num_kv_heads, q_len, head_dim)
            cache_key_states = mx.concatenate([past_key, cache_key_states], axis=2)
            cache_value_states = mx.concatenate([past_value, cache_value_states], axis=2)

        # Repeat key/value heads for grouped-query attention (for attention computation)
        if self.num_key_value_heads != self.num_attention_heads:
            key_states = self._repeat_kv(cache_key_states, self.num_key_value_groups)
            value_states = self._repeat_kv(cache_value_states, self.num_key_value_groups)
        else:
            key_states = cache_key_states
            value_states = cache_value_states

        # Use MLX fast attention for better performance
        # Prepare mask: scaled_dot_product_attention expects additive mask
        # Our attention_mask is already additive (with -inf for masked positions)
        attn_mask = None
        if attention_mask is not None:
            # attention_mask shape: (batch, 1, q_len, kv_len)
            # Slice to match actual key sequence length
            kv_len = key_states.shape[2]
            attn_mask = attention_mask[:, :, :, :kv_len]
            # scaled_dot_product_attention expects mask broadcastable to (batch, num_heads, q_len, kv_len)
            # Our mask is (batch, 1, q_len, kv_len), which will broadcast correctly
            # No need to squeeze - keep it as (batch, 1, q_len, kv_len) for proper broadcasting

        # Cast to float32 for numerical stability (scaled_dot_product_attention handles this internally)
        # But we'll cast inputs to float32 to match our previous behavior
        query_states_f32 = query_states.astype(mx.float32)
        key_states_f32 = key_states.astype(mx.float32)
        value_states_f32 = value_states.astype(mx.float32)

        # Use fast MLX attention
        # scaled_dot_product_attention expects (batch, num_heads, seq_len, head_dim)
        attn_output = scaled_dot_product_attention(
            query_states_f32,
            key_states_f32,
            value_states_f32,
            scale=self.scaling,
            mask=attn_mask,
        )

        # Cast back to original dtype
        attn_output = attn_output.astype(query_states.dtype)

        # Reshape back: (batch_size, seq_len, num_heads * head_dim)
        attn_output = attn_output.transpose(0, 2, 1, 3).reshape(bsz, q_len, self.num_attention_heads * self.head_dim)

        # Output projection
        attn_output = self.o_proj(attn_output)

        # Return cached key/value states (before GQA repetition)
        # cache_key_states and cache_value_states already include past cache if provided
        return attn_output, (cache_key_states, cache_value_states)

    @staticmethod
    def _repeat_kv(hidden_states: mx.array, n_rep: int) -> mx.array:
        shape = hidden_states.shape

        # If already repeated (batch, num_kv_heads, n_rep, seq_len, head_dim),
        # just collapse the repetition dimension.
        if len(shape) == 5:
            batch, num_key_value_heads, rep, slen, head_dim = shape
            return hidden_states.reshape(batch, num_key_value_heads * rep, slen, head_dim)

        batch, num_key_value_heads, slen, head_dim = shape
        hidden_states = mx.expand_dims(hidden_states, axis=2)
        hidden_states = mx.broadcast_to(hidden_states, (batch, num_key_value_heads, n_rep, slen, head_dim))
        return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)

    @staticmethod
    def _apply_rotary_pos_emb(
        q: mx.array,
        k: mx.array,
        cos: mx.array,
        sin: mx.array,
        unsqueeze_dim: int = 1,
    ) -> tuple[mx.array, mx.array]:
        # Unsqueeze cos/sin for broadcasting to q/k
        cos = mx.expand_dims(cos, axis=unsqueeze_dim)
        sin = mx.expand_dims(sin, axis=unsqueeze_dim)

        # RoPE application: q_embed = q * cos + rotate_half(q) * sin
        # Works fine in float16 for modern implementations
        q_embed = (q * cos) + (Qwen3VLAttention._rotate_half(q) * sin)
        k_embed = (k * cos) + (Qwen3VLAttention._rotate_half(k) * sin)

        return q_embed, k_embed

    @staticmethod
    def _apply_multimodal_rotary_pos_emb(
        q: mx.array,
        k: mx.array,
        position_embeddings: tuple[mx.array, mx.array],
        mrope_section: list[int],
        unsqueeze_dim: int = 1,
    ) -> tuple[mx.array, mx.array]:
        cos, sin = position_embeddings

        # Double the section sizes (for cos and sin)
        mrope_section_doubled = [s * 2 for s in mrope_section]

        # Split cos/sin into chunks for each section
        cos_chunks = []
        sin_chunks = []
        start_idx = 0
        for section_size in mrope_section_doubled:
            end_idx = start_idx + section_size
            cos_chunk = cos[..., start_idx:end_idx]
            sin_chunk = sin[..., start_idx:end_idx]
            cos_chunks.append(cos_chunk)
            sin_chunks.append(sin_chunk)
            start_idx = end_idx

        # Select chunks based on position (modulo 3 for 3 modalities)
        # This implements the multimodal RoPE selection logic
        cos_selected = [chunk[i % 3] for i, chunk in enumerate(cos_chunks)]
        sin_selected = [chunk[i % 3] for i, chunk in enumerate(sin_chunks)]

        # Concatenate selected chunks
        cos_combined = mx.concatenate(cos_selected, axis=-1)
        sin_combined = mx.concatenate(sin_selected, axis=-1)

        # Unsqueeze for broadcasting if needed
        if unsqueeze_dim == 1:
            cos_combined = mx.expand_dims(cos_combined, axis=1)
            sin_combined = mx.expand_dims(sin_combined, axis=1)

        # Apply RoPE: q_embed = q * cos + rotate_half(q) * sin
        # Works fine in float16 for modern implementations
        q_embed = (q * cos_combined) + (Qwen3VLAttention._rotate_half(q) * sin_combined)
        k_embed = (k * cos_combined) + (Qwen3VLAttention._rotate_half(k) * sin_combined)

        return q_embed, k_embed

    @staticmethod
    def _rotate_half(x: mx.array) -> mx.array:
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return mx.concatenate([-x2, x1], axis=-1)
