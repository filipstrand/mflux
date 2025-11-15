import mlx.core as mx
from mlx import nn
from mlx.core.fast import scaled_dot_product_attention


def apply_rotary_emb_mlx(
    x: mx.array,
    freqs_cos: mx.array,
    freqs_sin: mx.array,
) -> mx.array:
    """
    MLX port of diffusers.models.embeddings.apply_rotary_emb for the FIBO layout.

    Expects:
      - x: [B, S, H, D]
      - freqs_cos, freqs_sin: [S, D]
    """
    if x.ndim != 4:
        raise ValueError(f"Expected x with shape [B, S, H, D], got {x.shape}")

    bsz, seq_len, num_heads, head_dim = x.shape

    if freqs_cos.shape != (seq_len, head_dim) or freqs_sin.shape != (seq_len, head_dim):
        raise ValueError(
            f"Expected freqs_cos/freqs_sin with shape ({seq_len}, {head_dim}), "
            f"got {freqs_cos.shape} and {freqs_sin.shape}"
        )

    # Broadcast cos/sin to [B, S, H, D]
    cos = mx.expand_dims(mx.expand_dims(freqs_cos, axis=0), axis=2)
    sin = mx.expand_dims(mx.expand_dims(freqs_sin, axis=0), axis=2)

    # Split last dim into complex pairs
    x2 = x.reshape(bsz, seq_len, num_heads, -1, 2)
    x_real = x2[..., 0]
    x_imag = x2[..., 1]

    x_rotated_real = -x_imag
    x_rotated_imag = x_real

    x_rotated = mx.stack([x_rotated_real, x_rotated_imag], axis=-1).reshape(bsz, seq_len, num_heads, head_dim)

    out = (x.astype(mx.float32) * cos + x_rotated.astype(mx.float32) * sin).astype(x.dtype)
    return out


class FiboJointAttention(nn.Module):
    """
    MLX implementation of BriaFiboAttention for the joint (image + context) case.

    This mirrors the core math of diffusers.models.transformers.transformer_bria_fibo.BriaFiboAttention
    with BriaFiboAttnProcessor, but specialized for the FIBO configuration:
      - query_dim = dim
      - added_kv_proj_dim = dim
      - out_dim = dim
      - qk RMSNorm, no dropout in inference.
    """

    def __init__(self, dim: int, num_attention_heads: int, attention_head_dim: int, eps: float = 1e-6):
        super().__init__()
        self.head_dim = attention_head_dim
        self.num_heads = num_attention_heads
        self.inner_dim = dim

        self.to_q = nn.Linear(dim, self.inner_dim)
        self.to_k = nn.Linear(dim, self.inner_dim)
        self.to_v = nn.Linear(dim, self.inner_dim)

        self.norm_q = nn.RMSNorm(self.head_dim, eps=eps)
        self.norm_k = nn.RMSNorm(self.head_dim, eps=eps)

        # Added KV projections for encoder_hidden_states
        self.add_q_proj = nn.Linear(dim, self.inner_dim)
        self.add_k_proj = nn.Linear(dim, self.inner_dim)
        self.add_v_proj = nn.Linear(dim, self.inner_dim)

        self.norm_added_q = nn.RMSNorm(self.head_dim, eps=eps)
        self.norm_added_k = nn.RMSNorm(self.head_dim, eps=eps)

        # Output projections
        # Keep structure compatible with existing weight mapping which expects a list
        # (like Flux JointAttention: to_out[0] is the main linear projection).
        self.to_out = [nn.Linear(self.inner_dim, dim)]
        self.to_add_out = nn.Linear(self.inner_dim, dim)

    def __call__(
        self,
        hidden_states: mx.array,
        encoder_hidden_states: mx.array,
        image_rotary_emb: tuple[mx.array, mx.array],
        attention_mask: mx.array | None = None,
    ) -> tuple[mx.array, mx.array]:
        """
        Args:
            hidden_states: [B, S_img, D]
            encoder_hidden_states: [B, S_ctx, D]
            image_rotary_emb: (cos, sin) each [S_total, head_dim]
            attention_mask: [B, 1, S_total, S_total] additive mask (0 or -inf)

        Returns:
            attn_output: [B, S_img, D]
            context_attn_output: [B, S_ctx, D]
        """
        batch_size, seq_img, dim = hidden_states.shape
        _, seq_ctx, _ = encoder_hidden_states.shape

        cos, sin = image_rotary_emb

        # QKV for image stream: [B, S_img, inner_dim]
        query = self.to_q(hidden_states)
        key = self.to_k(hidden_states)
        value = self.to_v(hidden_states)

        # QKV for context stream (added_kv): [B, S_ctx, inner_dim]
        enc_query = self.add_q_proj(encoder_hidden_states)
        enc_key = self.add_k_proj(encoder_hidden_states)
        enc_value = self.add_v_proj(encoder_hidden_states)

        # Reshape to [B, S, H, D]
        def _reshape_bshd(x: mx.array, seq_len: int) -> mx.array:
            return mx.reshape(x, (batch_size, seq_len, self.num_heads, self.head_dim))

        query = _reshape_bshd(query, seq_img)
        key = _reshape_bshd(key, seq_img)
        value = _reshape_bshd(value, seq_img)

        enc_query = _reshape_bshd(enc_query, seq_ctx)
        enc_key = _reshape_bshd(enc_key, seq_ctx)
        enc_value = _reshape_bshd(enc_value, seq_ctx)

        # RMSNorm over last dim: produce normalized Q/K matching BriaFiboAttention semantics.
        query = self.norm_q(query.astype(mx.float32)).astype(query.dtype)
        key = self.norm_k(key.astype(mx.float32)).astype(key.dtype)

        enc_query = self.norm_added_q(enc_query.astype(mx.float32)).astype(enc_query.dtype)
        enc_key = self.norm_added_k(enc_key.astype(mx.float32)).astype(enc_key.dtype)

        # Concatenate encoder + image along sequence dim (matches BriaFiboAttnProcessor)
        query = mx.concatenate([enc_query, query], axis=1)
        key = mx.concatenate([enc_key, key], axis=1)
        value = mx.concatenate([enc_value, value], axis=1)

        seq_total = seq_ctx + seq_img

        # Apply RoPE to Q,K (layout [B, S_total, H, D])
        query = apply_rotary_emb_mlx(query, cos, sin)
        key = apply_rotary_emb_mlx(key, cos, sin)

        # ===== Manual attention core (BriaFibo-style) =====
        # Convert to [B, H, S, D]
        query_bhsd = mx.transpose(query, (0, 2, 1, 3))
        key_bhsd = mx.transpose(key, (0, 2, 1, 3))
        value_bhsd = mx.transpose(value, (0, 2, 1, 3))

        # Scaled dot-product attention with additive mask (same semantics as PyTorch).
        scale = 1.0 / mx.sqrt(mx.array(self.head_dim, dtype=query_bhsd.dtype))
        attn_scores = (
            mx.matmul(
                query_bhsd,
                mx.transpose(key_bhsd, (0, 1, 3, 2)),
            )
            * scale
        )

        if attention_mask is not None:
            # attention_mask saved from PyTorch as [B, 1, S_total, S_total] with 0 or large negative values.
            mask = attention_mask.astype(attn_scores.dtype)
            attn_scores = attn_scores + mask

        attn_weights = mx.softmax(attn_scores, axis=-1)
        attn_output_bhsd = mx.matmul(attn_weights, value_bhsd)

        # Back to [B, S_total, H, D] then flatten to [B, S_total, inner_dim]
        attn_output = mx.transpose(attn_output_bhsd, (0, 2, 1, 3))
        attn_output = mx.reshape(attn_output, (batch_size, seq_total, self.inner_dim))

        # Split back into context and image streams
        context_attn_output = attn_output[:, :seq_ctx, :]
        hidden_attn_output = attn_output[:, seq_ctx:, :]

        # Output projections
        hidden_attn_output = self.to_out[0](hidden_attn_output)
        context_attn_output = self.to_add_out(context_attn_output)

        return hidden_attn_output, context_attn_output


class FiboSingleAttention(nn.Module):
    """
    Self-attention used in BriaFiboSingleTransformerBlock.

    Mirrors the core math of diffusers' `Attention` configured with:
        qk_norm = "rms_norm", pre_only = True, no dropout.
    """

    def __init__(self, dim: int, num_attention_heads: int, attention_head_dim: int, eps: float = 1e-6):
        super().__init__()
        self.head_dim = attention_head_dim
        self.num_heads = num_attention_heads
        self.inner_dim = dim

        self.to_q = nn.Linear(dim, self.inner_dim)
        self.to_k = nn.Linear(dim, self.inner_dim)
        self.to_v = nn.Linear(dim, self.inner_dim)

        self.norm_q = nn.RMSNorm(self.head_dim, eps=eps)
        self.norm_k = nn.RMSNorm(self.head_dim, eps=eps)

    def __call__(
        self,
        hidden_states: mx.array,
        image_rotary_emb: tuple[mx.array, mx.array],
        attention_mask: mx.array | None = None,
    ) -> mx.array:
        """
        Args:
            hidden_states: [B, S, D]
            image_rotary_emb: (cos, sin) each [S, head_dim]
            attention_mask: [B, 1, S, S] or None
        Returns:
            attn_output: [B, S, D]
        """
        batch_size, seq_len, _ = hidden_states.shape
        cos, sin = image_rotary_emb

        # [B, S, inner_dim]
        query = self.to_q(hidden_states)
        key = self.to_k(hidden_states)
        value = self.to_v(hidden_states)

        # [B, S, H, D]
        def _reshape_bshd(x: mx.array) -> mx.array:
            return mx.reshape(x, (batch_size, seq_len, self.num_heads, self.head_dim))

        query = _reshape_bshd(query)
        key = _reshape_bshd(key)
        value = _reshape_bshd(value)

        # RMSNorm
        query = self.norm_q(query.astype(mx.float32)).astype(query.dtype)
        key = self.norm_k(key.astype(mx.float32)).astype(key.dtype)

        # RoPE
        query = apply_rotary_emb_mlx(query, cos, sin)
        key = apply_rotary_emb_mlx(key, cos, sin)

        # [B, H, S, D]
        query_bhsd = mx.transpose(query, (0, 2, 1, 3))
        key_bhsd = mx.transpose(key, (0, 2, 1, 3))
        value_bhsd = mx.transpose(value, (0, 2, 1, 3))

        scale = 1.0 / mx.sqrt(mx.array(self.head_dim, dtype=query_bhsd.dtype))

        attn_output = scaled_dot_product_attention(
            query_bhsd,
            key_bhsd,
            value_bhsd,
            scale=scale,
            mask=attention_mask,
        )

        # [B, S, H, D] -> [B, S, inner_dim]
        attn_output = mx.transpose(attn_output, (0, 2, 1, 3))
        attn_output = mx.reshape(attn_output, (batch_size, seq_len, self.inner_dim))
        return attn_output
