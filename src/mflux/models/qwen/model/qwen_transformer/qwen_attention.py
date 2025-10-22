from __future__ import annotations

import mlx.core as mx
from mlx import nn

from mflux.models.flux.model.flux_transformer.common.attention_utils import AttentionUtils


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
        # Match PyTorch QwenDoubleStreamAttnProcessor2_0 exactly (lines 296-337)

        # 1a. Compute QKV for image stream (sample projections)
        # PyTorch: img_query = attn.to_q(hidden_states)
        img_query = self.to_q(img_modulated)
        img_key = self.to_k(img_modulated)
        img_value = self.to_v(img_modulated)

        # 1b. Compute QKV for text stream (context projections)
        # PyTorch: txt_query = attn.add_q_proj(encoder_hidden_states)
        txt_query = self.add_q_proj(txt_modulated)
        txt_key = self.add_k_proj(txt_modulated)
        txt_value = self.add_v_proj(txt_modulated)

        # 2. Reshape for multi-head attention
        # PyTorch: img_query = img_query.unflatten(-1, (attn.heads, -1))
        # unflatten(-1, (heads, -1)) reshapes [B, S, H*D] -> [B, S, H, D]
        img_query = mx.reshape(img_query, (img_query.shape[0], img_query.shape[1], self.num_heads, self.head_dim))
        img_key = mx.reshape(img_key, (img_key.shape[0], img_key.shape[1], self.num_heads, self.head_dim))
        img_value = mx.reshape(img_value, (img_value.shape[0], img_value.shape[1], self.num_heads, self.head_dim))

        txt_query = mx.reshape(txt_query, (txt_query.shape[0], txt_query.shape[1], self.num_heads, self.head_dim))
        txt_key = mx.reshape(txt_key, (txt_key.shape[0], txt_key.shape[1], self.num_heads, self.head_dim))
        txt_value = mx.reshape(txt_value, (txt_value.shape[0], txt_value.shape[1], self.num_heads, self.head_dim))

        # 3. Apply QK normalization
        # PyTorch: if attn.norm_q is not None: img_query = attn.norm_q(img_query)
        if self.norm_q is not None:
            img_query = self.norm_q(img_query)
        if self.norm_k is not None:
            img_key = self.norm_k(img_key)
        if self.norm_added_q is not None:
            txt_query = self.norm_added_q(txt_query)
        if self.norm_added_k is not None:
            txt_key = self.norm_added_k(txt_key)

        # 4. Apply RoPE
        # PyTorch: if image_rotary_emb is not None: img_query = apply_rotary_emb_qwen(img_query, img_freqs, use_real=False)
        if image_rotary_emb is not None:
            # MLX uses (cos, sin) tuples, PyTorch uses complex freqs directly
            # image_rotary_emb is a tuple: (img_rotary_emb, txt_rotary_emb) where each is (cos, sin)
            (img_cos, img_sin), (txt_cos, txt_sin) = image_rotary_emb
            img_query = QwenAttention._apply_rope_qwen(img_query, img_cos, img_sin)
            img_key = QwenAttention._apply_rope_qwen(img_key, img_cos, img_sin)
            txt_query = QwenAttention._apply_rope_qwen(txt_query, txt_cos, txt_sin)
            txt_key = QwenAttention._apply_rope_qwen(txt_key, txt_cos, txt_sin)

        # 5. Concatenate for joint attention
        # PyTorch: joint_query = torch.cat([txt_query, img_query], dim=1)
        # Order: [text, image]
        joint_query = mx.concatenate([txt_query, img_query], axis=1)
        joint_key = mx.concatenate([txt_key, img_key], axis=1)
        joint_value = mx.concatenate([txt_value, img_value], axis=1)

        # 6. Compute joint attention
        # PyTorch: scale_value = 1.0 / (joint_query.shape[-1] ** 0.5)
        # PyTorch: joint_hidden_states = dispatch_attention_fn(...)
        # PyTorch uses [B, S, H, D] format natively (no transpose needed)
        seq_txt = txt_modulated.shape[1]

        # Create mask for [B,S,H,D] format
        mask = self._convert_mask_for_qwen(
            mask=encoder_hidden_states_mask,
            joint_seq_len=joint_query.shape[1],
            txt_seq_len=seq_txt,
        )

        # Compute attention - Qwen uses [B,S,H,D] format
        hidden_states = self._compute_attention_qwen(
            query=joint_query,
            key=joint_key,
            value=joint_value,
            mask=mask,
            block_idx=block_idx,
        )

        # 7. Reshape back and convert dtype
        # PyTorch: joint_hidden_states = joint_hidden_states.flatten(2, 3)
        # PyTorch: joint_hidden_states = joint_hidden_states.to(joint_query.dtype)
        # (Already flattened in _compute_attention_qwen, dtype already matches)

        # 8. Split attention outputs back
        # PyTorch: txt_attn_output = joint_hidden_states[:, :seq_txt, :]
        # PyTorch: img_attn_output = joint_hidden_states[:, seq_txt:, :]
        txt_attn_output = hidden_states[:, :seq_txt, :]
        img_attn_output = hidden_states[:, seq_txt:, :]

        # 🔧 LOAD: Load PyTorch's SPLIT SDPA outputs for timestep 0 (after split, before projection)
        import os

        LOAD_SDPA_TIMESTEP0 = os.environ.get("LOAD_SDPA_TIMESTEP0", "0") == "1"
        if LOAD_SDPA_TIMESTEP0 and block_idx is not None:
            import numpy as np

            sdpa_dir = "/Users/filip/Desktop/pytorch_sdpa_timestep0"

            # Get prefix (pos/neg) from environment variable
            prefix = os.environ.get("SDPA_LOAD_PREFIX", "pos")

            img_path = f"{sdpa_dir}/block{block_idx:02d}_{prefix}_img_sdpa_output.npy"
            txt_path = f"{sdpa_dir}/block{block_idx:02d}_{prefix}_txt_sdpa_output.npy"

            if os.path.exists(img_path) and os.path.exists(txt_path):
                # Load PyTorch SDPA split outputs [B, S, H, D] and reshape to [B, S, H*D]
                loaded_img = mx.array(np.load(img_path))  # [B, S_img, H, D]
                loaded_txt = mx.array(np.load(txt_path))  # [B, S_txt, H, D]

                batch_size, seq_len_img, num_heads, head_dim = loaded_img.shape
                _, seq_len_txt, _, _ = loaded_txt.shape

                img_attn_output = mx.reshape(loaded_img, (batch_size, seq_len_img, num_heads * head_dim))
                txt_attn_output = mx.reshape(loaded_txt, (batch_size, seq_len_txt, num_heads * head_dim))

        # 9. Apply output projections
        # PyTorch: img_attn_output = attn.to_out[0](img_attn_output)
        img_attn_output = self.attn_to_out[0](img_attn_output)
        # PyTorch: if len(attn.to_out) > 1: img_attn_output = attn.to_out[1](img_attn_output)  # dropout
        # Note: MLX doesn't have dropout in the list, so we skip this

        # PyTorch: txt_attn_output = attn.to_add_out(txt_attn_output)
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
        """
        Compute attention for Qwen (matches PyTorch dispatch_attention_fn).

        PyTorch code (lines 340-356):
            scale_value = 1.0 / (joint_query.shape[-1] ** 0.5)
            joint_hidden_states = dispatch_attention_fn(
                joint_query, joint_key, joint_value,
                attn_mask=attention_mask, dropout_p=0.0, is_causal=False, ...
            )
            joint_hidden_states = joint_hidden_states.flatten(2, 3)
            joint_hidden_states = joint_hidden_states.to(joint_query.dtype)

        Input: [B, S, H, D] (PyTorch format, no transpose needed)
        Output: [B, S, H*D] (flattened)
        """
        from mlx.core.fast import scaled_dot_product_attention

        # PyTorch uses [B, S, H, D] format natively (no transpose needed)
        # But MLX's scaled_dot_product_attention expects [B, H, S, D]
        # So we need to transpose for MLX's attention function
        query_bhsd = mx.transpose(query, (0, 2, 1, 3))  # [B, S, H, D] -> [B, H, S, D]
        key_bhsd = mx.transpose(key, (0, 2, 1, 3))  # [B, S, H, D] -> [B, H, S, D]
        value_bhsd = mx.transpose(value, (0, 2, 1, 3))  # [B, S, H, D] -> [B, H, S, D]

        # PyTorch: scale_value = 1.0 / (joint_query.shape[-1] ** 0.5)
        # head_dim is the last dimension of query in [B, S, H, D] format
        head_dim = query.shape[-1]
        scale_value = 1.0 / (head_dim**0.5)

        # PyTorch: dispatch_attention_fn uses scaled_dot_product_attention internally
        # MLX: use scaled_dot_product_attention directly
        # Note: mask needs to be in [B, H, S, S] format for MLX if provided
        hidden_states_bhsd = scaled_dot_product_attention(
            query_bhsd, key_bhsd, value_bhsd, scale=scale_value, mask=mask
        )  # [B, H, S, D]

        # Transpose back: [B, H, S, D] -> [B, S, H, D]
        hidden_states = mx.transpose(hidden_states_bhsd, (0, 2, 1, 3))  # [B, S, H, D]

        # PyTorch: joint_hidden_states.flatten(2, 3) - flattens dimensions 2 and 3 (H and D)
        # [B, S, H, D] -> [B, S, H*D]
        batch_size = hidden_states.shape[0]
        seq_len = hidden_states.shape[1]
        hidden_states = mx.reshape(hidden_states, (batch_size, seq_len, self.num_heads * self.head_dim))

        # PyTorch: joint_hidden_states.to(joint_query.dtype)
        # Ensure dtype matches query dtype
        hidden_states = hidden_states.astype(query.dtype)

        return hidden_states

    @staticmethod
    def _convert_mask_for_qwen(
        mask: mx.array | None,
        joint_seq_len: int,
        txt_seq_len: int,
    ) -> mx.array | None:
        """
        Convert key padding mask for attention.

        Since we transpose to [B,H,S,D] for MLX's scaled_dot_product_attention,
        we need mask shape [B,1,1,S] (same as Flux).
        """
        if mask is None:
            return None

        bsz = mask.shape[0]
        img_seq_len = joint_seq_len - txt_seq_len

        # Create joint mask: [text_mask, image_ones]
        ones_img = mx.ones((bsz, img_seq_len), dtype=mx.float32)
        joint_mask = mx.concatenate([mask.astype(mx.float32), ones_img], axis=1)

        # Check if mask is all ones (no actual masking needed)
        # This avoids passing a zero mask to SDPA, which might behave differently than None
        if mx.all(joint_mask >= 0.999):  # Use >= 0.999 to handle floating point imprecision
            return None

        # Convert to additive mask for scaled_dot_product_attention
        # Shape: [B,1,1,S] for [B,H,S,D] format
        additive = (1.0 - joint_mask) * (-1e9)
        return additive.reshape((additive.shape[0], 1, 1, additive.shape[1]))

    @staticmethod
    def _apply_rotary_embeddings_joint(
        joint_q: mx.array,
        joint_k: mx.array,
        txt_seq_len: int,
        image_rotary_emb: tuple[tuple[mx.array, mx.array], tuple[mx.array, mx.array]],
    ) -> tuple[mx.array, mx.array]:
        """
        Apply RoPE to joint Q,K tensors in [B,S,H,D] format using complex rotation.

        Matches Diffusers' complex rotation approach (use_real=False).
        RoPE embeddings are tuples of (cos, sin) tensors with shape [S, D/2].
        """
        (img_cos, img_sin), (txt_cos, txt_sin) = image_rotary_emb

        # Extract separate parts from [B,S,H,D] - split on sequence dimension (axis=1)
        txt_q = joint_q[:, :txt_seq_len, :, :]
        img_q = joint_q[:, txt_seq_len:, :, :]
        txt_k = joint_k[:, :txt_seq_len, :, :]
        img_k = joint_k[:, txt_seq_len:, :, :]

        # Apply RoPE using complex rotation (matches Diffusers use_real=False)
        img_q = QwenAttention._apply_rope_complex(img_q, img_cos, img_sin)
        img_k = QwenAttention._apply_rope_complex(img_k, img_cos, img_sin)
        txt_q = QwenAttention._apply_rope_complex(txt_q, txt_cos, txt_sin)
        txt_k = QwenAttention._apply_rope_complex(txt_k, txt_cos, txt_sin)

        # Concatenate back [text, image] on sequence dimension (axis=1)
        joint_q = mx.concatenate([txt_q, img_q], axis=1)
        joint_k = mx.concatenate([txt_k, img_k], axis=1)

        return joint_q, joint_k

    @staticmethod
    def _apply_rope_qwen(x: mx.array, cos_vals: mx.array, sin_vals: mx.array) -> mx.array:
        """
        Apply RoPE using complex number multiplication (matches PyTorch apply_rotary_emb_qwen with use_real=False).

        PyTorch code (lines 142-147):
            x_rotated = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
            freqs_cis = freqs_cis.unsqueeze(1)
            x_out = torch.view_as_real(x_rotated * freqs_cis).flatten(3)
            return x_out.type_as(x)

        In PyTorch, freqs_cis is a complex tensor e^(i*θ) = cos(θ) + i*sin(θ).
        In MLX, we receive cos and sin separately.

        Args:
            x: Input tensor [B, S, H, D]
            cos_vals: Cosine frequencies [S, D] (where D matches head_dim)
            sin_vals: Sine frequencies [S, D]

        Returns:
            Rotated tensor [B, S, H, D]
        """
        # Step 1: Reshape x to treat last dimension as complex pairs: [B, S, H, D] -> [B, S, H, D//2, 2]
        # PyTorch: x.float().reshape(*x.shape[:-1], -1, 2)
        x_float = x.astype(mx.float32)
        x_reshaped = mx.reshape(x_float, (*x.shape[:-1], -1, 2))  # [B, S, H, D//2, 2]

        # Step 2: Extract real and imaginary parts
        x_real = x_reshaped[..., 0]  # [B, S, H, D//2]
        x_imag = x_reshaped[..., 1]  # [B, S, H, D//2]

        # Step 3: Prepare freqs_cis for broadcasting
        # PyTorch: freqs_cis.unsqueeze(1) -> [S, 1, D]
        # For MLX, we need [1, S, 1, D//2] to broadcast with [B, S, H, D//2]
        # cos_vals and sin_vals are [S, D] where D = head_dim
        # We need to reshape to [1, S, 1, D//2] (assuming D matches head_dim)
        freqs_cos = cos_vals[None, :, None, :]  # [1, S, 1, D] -> but we need D//2
        freqs_sin = sin_vals[None, :, None, :]  # [1, S, 1, D]

        # Actually, cos_vals/sin_vals should be [S, D//2] if they match the head_dim structure
        # But let's check - if D = head_dim, then we need D//2 for the complex pairs
        # Let's assume cos_vals/sin_vals are [S, head_dim] and we need to split to [S, head_dim//2]
        if freqs_cos.shape[-1] != x_real.shape[-1]:
            # If dimensions don't match, assume cos/sin are full head_dim and we need to take first half
            freqs_cos = freqs_cos[..., : x_real.shape[-1]]  # [1, S, 1, D//2]
            freqs_sin = freqs_sin[..., : x_real.shape[-1]]  # [1, S, 1, D//2]

        # Step 4: Complex multiplication: (x_real + i*x_imag) * (freqs_cos + i*freqs_sin)
        # = (x_real*freqs_cos - x_imag*freqs_sin) + i*(x_real*freqs_sin + x_imag*freqs_cos)
        out_real = x_real * freqs_cos - x_imag * freqs_sin  # [B, S, H, D//2]
        out_imag = x_real * freqs_sin + x_imag * freqs_cos  # [B, S, H, D//2]

        # Step 5: Stack real and imag parts: [B, S, H, D//2, 2]
        # PyTorch: torch.view_as_real(...) -> [B, S, H, D//2, 2]
        out_pairs = mx.stack([out_real, out_imag], axis=-1)  # [B, S, H, D//2, 2]

        # Step 6: Flatten: [B, S, H, D//2, 2] -> [B, S, H, D]
        # PyTorch: .flatten(3) flattens dimensions starting at index 3
        x_out = mx.reshape(out_pairs, (*x.shape[:-1], -1))  # [B, S, H, D]

        # Step 7: Convert back to original dtype
        # PyTorch: .type_as(x)
        return x_out.astype(x.dtype)

    @staticmethod
    def _apply_rotary_embeddings_separate(
        q: mx.array,
        k: mx.array,
        rotary_emb: tuple[mx.array, mx.array],
    ) -> tuple[mx.array, mx.array]:
        """Alternative RoPE application for Edit model - applies RoPE before concatenation"""
        # rotary_emb is now a tuple of (cos, sin) with shape [S, D/2]
        cos, sin = rotary_emb

        # Convert to [B, S, H, D] format for RoPE application
        q_bshd = mx.transpose(q, (0, 2, 1, 3))
        k_bshd = mx.transpose(k, (0, 2, 1, 3))

        seq_len = q_bshd.shape[1]

        # Ensure RoPE dimensions match sequence length
        if cos.shape[0] != seq_len:
            if cos.shape[0] < seq_len:
                # Pad with the last embedding
                pad_len = seq_len - cos.shape[0]
                last_cos = mx.tile(cos[-1:], (pad_len, 1))
                last_sin = mx.tile(sin[-1:], (pad_len, 1))
                cos = mx.concatenate([cos, last_cos], axis=0)
                sin = mx.concatenate([sin, last_sin], axis=0)
            else:
                # Truncate to sequence length
                cos = cos[:seq_len]
                sin = sin[:seq_len]

        # Apply RoPE
        q_bshd, k_bshd = AttentionUtils.apply_rope_bshd(q_bshd, k_bshd, cos, sin)

        # Convert back to [B, H, S, D] format
        q = mx.transpose(q_bshd, (0, 2, 1, 3))
        k = mx.transpose(k_bshd, (0, 2, 1, 3))

        return q, k
