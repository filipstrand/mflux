from __future__ import annotations

import mlx.core as mx
from mlx import nn


class QwenAttention(nn.Module):
    def __init__(self, dim: int = 3072, num_heads: int = 24, head_dim: int = 128):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = head_dim

        # Attention projections for image stream
        self.to_q = nn.Linear(dim, dim)
        self.to_k = nn.Linear(dim, dim)
        self.to_v = nn.Linear(dim, dim)

        # Attention projections for text stream
        self.add_q_proj = nn.Linear(dim, dim)
        self.add_k_proj = nn.Linear(dim, dim)
        self.add_v_proj = nn.Linear(dim, dim)

        # Query/Key normalization
        self.norm_q = nn.RMSNorm(self.head_dim, eps=1e-6)
        self.norm_k = nn.RMSNorm(self.head_dim, eps=1e-6)
        self.norm_added_q = nn.RMSNorm(self.head_dim, eps=1e-6)
        self.norm_added_k = nn.RMSNorm(self.head_dim, eps=1e-6)

        # Output projections
        self.attn_to_out = [nn.Linear(dim, dim)]
        self.to_add_out = nn.Linear(dim, dim)

    def __call__(
        self,
        img_modulated: mx.array,
        txt_modulated: mx.array,
        encoder_hidden_states_mask: mx.array | None,
        image_rotary_emb: tuple[mx.array, mx.array],
        block_idx: int | None = None,  # For debugging context
    ) -> tuple[mx.array, mx.array]:
        # 1a. Compute QKV for image stream
        img_query = self.to_q(img_modulated)
        img_key = self.to_k(img_modulated)
        img_value = self.to_v(img_modulated)

        # 1b. Compute QKV for text stream
        txt_query = self.add_q_proj(txt_modulated)
        txt_key = self.add_k_proj(txt_modulated)
        txt_value = self.add_v_proj(txt_modulated)

        # 2. Reshape for multi-head attention: [B, S, H*D] -> [B, S, H, D]
        img_query = mx.reshape(img_query, (img_query.shape[0], img_query.shape[1], self.num_heads, self.head_dim))
        img_key = mx.reshape(img_key, (img_key.shape[0], img_key.shape[1], self.num_heads, self.head_dim))
        img_value = mx.reshape(img_value, (img_value.shape[0], img_value.shape[1], self.num_heads, self.head_dim))

        txt_query = mx.reshape(txt_query, (txt_query.shape[0], txt_query.shape[1], self.num_heads, self.head_dim))
        txt_key = mx.reshape(txt_key, (txt_key.shape[0], txt_key.shape[1], self.num_heads, self.head_dim))
        txt_value = mx.reshape(txt_value, (txt_value.shape[0], txt_value.shape[1], self.num_heads, self.head_dim))

        # 3. Apply QK normalization
        if self.norm_q is not None:
            img_query = self.norm_q(img_query)
        if self.norm_k is not None:
            img_key = self.norm_k(img_key)
        if self.norm_added_q is not None:
            txt_query = self.norm_added_q(txt_query)
        if self.norm_added_k is not None:
            txt_key = self.norm_added_k(txt_key)

        # 4. Apply RoPE
        if image_rotary_emb is not None:
            (img_cos, img_sin), (txt_cos, txt_sin) = image_rotary_emb
            img_query = QwenAttention._apply_rope_qwen(img_query, img_cos, img_sin)
            img_key = QwenAttention._apply_rope_qwen(img_key, img_cos, img_sin)
            txt_query = QwenAttention._apply_rope_qwen(txt_query, txt_cos, txt_sin)
            txt_key = QwenAttention._apply_rope_qwen(txt_key, txt_cos, txt_sin)

        # 5. Concatenate for joint attention: [text, image]
        joint_query = mx.concatenate([txt_query, img_query], axis=1)
        joint_key = mx.concatenate([txt_key, img_key], axis=1)
        joint_value = mx.concatenate([txt_value, img_value], axis=1)

        # 6. Compute joint attention
        seq_txt = txt_modulated.shape[1]
        mask = self._convert_mask_for_qwen(
            mask=encoder_hidden_states_mask,
            joint_seq_len=joint_query.shape[1],
            txt_seq_len=seq_txt,
        )

        hidden_states = self._compute_attention_qwen(
            query=joint_query,
            key=joint_key,
            value=joint_value,
            mask=mask,
            block_idx=block_idx,
        )

        # 7. Split attention outputs back
        txt_attn_output = hidden_states[:, :seq_txt, :]
        img_attn_output = hidden_states[:, seq_txt:, :]

        # 🔧 LOAD: Load PyTorch's SPLIT SDPA outputs for timestep 0 (after split, before projection)
        import os

        LOAD_SDPA_TIMESTEP0 = os.environ.get("LOAD_SDPA_TIMESTEP0", "0") == "1"
        if LOAD_SDPA_TIMESTEP0 and block_idx is not None:
            import numpy as np

            sdpa_dir = "/Users/filip/Desktop/pytorch_sdpa_timestep0"
            prefix = os.environ.get("SDPA_LOAD_PREFIX", "pos")

            img_path = f"{sdpa_dir}/block{block_idx:02d}_{prefix}_img_sdpa_output.npy"
            txt_path = f"{sdpa_dir}/block{block_idx:02d}_{prefix}_txt_sdpa_output.npy"

            if os.path.exists(img_path) and os.path.exists(txt_path):
                loaded_img = mx.array(np.load(img_path))
                loaded_txt = mx.array(np.load(txt_path))

                batch_size, seq_len_img, num_heads, head_dim = loaded_img.shape
                _, seq_len_txt, _, _ = loaded_txt.shape

                img_attn_output = mx.reshape(loaded_img, (batch_size, seq_len_img, num_heads * head_dim))
                txt_attn_output = mx.reshape(loaded_txt, (batch_size, seq_len_txt, num_heads * head_dim))

        # 8. Apply output projections
        img_attn_output = self.attn_to_out[0](img_attn_output)
        txt_attn_output = self.to_add_out(txt_attn_output)

        return img_attn_output, txt_attn_output

    def _compute_attention_qwen(
        self,
        query: mx.array,
        key: mx.array,
        value: mx.array,
        mask: mx.array | None = None,
        block_idx: int | None = None,
    ) -> mx.array:
        from mlx.core.fast import scaled_dot_product_attention

        # Transpose [B, S, H, D] -> [B, H, S, D] for MLX's attention function
        query_bhsd = mx.transpose(query, (0, 2, 1, 3))
        key_bhsd = mx.transpose(key, (0, 2, 1, 3))
        value_bhsd = mx.transpose(value, (0, 2, 1, 3))

        head_dim = query.shape[-1]
        scale_value = 1.0 / (head_dim**0.5)

        hidden_states_bhsd = scaled_dot_product_attention(
            query_bhsd, key_bhsd, value_bhsd, scale=scale_value, mask=mask
        )

        # Transpose back: [B, H, S, D] -> [B, S, H, D]
        hidden_states = mx.transpose(hidden_states_bhsd, (0, 2, 1, 3))

        # Flatten: [B, S, H, D] -> [B, S, H*D]
        batch_size = hidden_states.shape[0]
        seq_len = hidden_states.shape[1]
        hidden_states = mx.reshape(hidden_states, (batch_size, seq_len, self.num_heads * self.head_dim))

        hidden_states = hidden_states.astype(query.dtype)

        return hidden_states

    @staticmethod
    def _convert_mask_for_qwen(
        mask: mx.array | None,
        joint_seq_len: int,
        txt_seq_len: int,
    ) -> mx.array | None:
        if mask is None:
            return None

        bsz = mask.shape[0]
        img_seq_len = joint_seq_len - txt_seq_len

        ones_img = mx.ones((bsz, img_seq_len), dtype=mx.float32)
        joint_mask = mx.concatenate([mask.astype(mx.float32), ones_img], axis=1)

        if mx.all(joint_mask >= 0.999):
            return None

        additive = (1.0 - joint_mask) * (-1e9)
        return additive.reshape((additive.shape[0], 1, 1, additive.shape[1]))

    @staticmethod
    def _apply_rope_qwen(x: mx.array, cos_vals: mx.array, sin_vals: mx.array) -> mx.array:
        # Reshape to complex pairs: [B, S, H, D] -> [B, S, H, D//2, 2]
        x_float = x.astype(mx.float32)
        x_reshaped = mx.reshape(x_float, (*x.shape[:-1], -1, 2))

        x_real = x_reshaped[..., 0]
        x_imag = x_reshaped[..., 1]

        # Prepare freqs for broadcasting: [1, S, 1, D//2]
        freqs_cos = cos_vals[None, :, None, :]
        freqs_sin = sin_vals[None, :, None, :]

        if freqs_cos.shape[-1] != x_real.shape[-1]:
            freqs_cos = freqs_cos[..., : x_real.shape[-1]]
            freqs_sin = freqs_sin[..., : x_real.shape[-1]]

        # Complex multiplication
        out_real = x_real * freqs_cos - x_imag * freqs_sin
        out_imag = x_real * freqs_sin + x_imag * freqs_cos

        # Stack and flatten: [B, S, H, D//2, 2] -> [B, S, H, D]
        out_pairs = mx.stack([out_real, out_imag], axis=-1)
        x_out = mx.reshape(out_pairs, (*x.shape[:-1], -1))

        return x_out.astype(x.dtype)
