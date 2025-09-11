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
    ) -> tuple[mx.array, mx.array]:
        # 1a. Compute Q,K,V for image stream (hidden_states)
        img_q, img_k, img_v = AttentionUtils.process_qkv(
            hidden_states=img_modulated,
            to_q=self.to_q,
            to_k=self.to_k,
            to_v=self.to_v,
            norm_q=self.norm_q,
            norm_k=self.norm_k,
            num_heads=self.num_heads,
            head_dim=self.head_dim,
        )

        # 1b. Compute Q,K,V for text stream (encoder_hidden_states)
        txt_q, txt_k, txt_v = AttentionUtils.process_qkv(
            hidden_states=txt_modulated,
            to_q=self.add_q_proj,
            to_k=self.add_k_proj,
            to_v=self.add_v_proj,
            norm_q=self.norm_added_q,
            norm_k=self.norm_added_k,
            num_heads=self.num_heads,
            head_dim=self.head_dim,
        )

        # 1c. Apply RoPE to separate streams first (like Diffusers)
        img_rotary_emb, txt_rotary_emb = image_rotary_emb

        # Apply RoPE to image stream
        img_q, img_k = QwenAttention._apply_rotary_embeddings_separate(
            q=img_q,
            k=img_k,
            rotary_emb=img_rotary_emb,
        )

        # Apply RoPE to text stream
        txt_q, txt_k = QwenAttention._apply_rotary_embeddings_separate(
            q=txt_q,
            k=txt_k,
            rotary_emb=txt_rotary_emb,
        )

        # 1d. Concatenate results [text, image] after RoPE application
        joint_q = mx.concatenate([txt_q, img_q], axis=2)
        joint_k = mx.concatenate([txt_k, img_k], axis=2)
        joint_v = mx.concatenate([txt_v, img_v], axis=2)

        # 2. Compute attention with optional masking
        mask = AttentionUtils.convert_key_padding_mask_to_additive_mask(
            mask=encoder_hidden_states_mask,
            joint_seq_len=joint_q.shape[2],
            txt_seq_len=txt_q.shape[2],
        )

        # 3. Compute attention
        hidden_states = AttentionUtils.compute_attention(
            query=joint_q,
            key=joint_k,
            value=joint_v,
            batch_size=1,
            num_heads=self.num_heads,
            head_dim=self.head_dim,
            mask=mask,
        )

        # 4. Separate the results
        txt_seq_len = txt_modulated.shape[1]
        txt_attn_output = hidden_states[:, :txt_seq_len]
        img_attn_output = hidden_states[:, txt_seq_len:]

        # 5. Project the output (Flux-style)
        img_attn_output = self.attn_to_out[0](img_attn_output.astype(mx.float32)).astype(img_modulated.dtype)
        txt_attn_output = self.to_add_out(txt_attn_output.astype(mx.float32)).astype(txt_modulated.dtype)

        return img_attn_output, txt_attn_output

    @staticmethod
    def _apply_rotary_embeddings_separate(
        q: mx.array,
        k: mx.array,
        rotary_emb: mx.array,
    ) -> tuple[mx.array, mx.array]:
        # Extract cos/sin from rotation matrix format [1, 1, S, D/2, 2, 2]
        cos = rotary_emb[0, 0, :, :, 0, 0]  # [S, D/2]
        sin = rotary_emb[0, 0, :, :, 1, 0]  # [S, D/2]

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
