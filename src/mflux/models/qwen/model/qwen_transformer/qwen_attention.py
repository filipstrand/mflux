from __future__ import annotations

import mlx.core as mx
from mlx import nn

from mflux.models.flux.model.flux_transformer.common.attention_utils import AttentionUtils
from mflux.models.qwen.model.qwen_transformer.custom_rms_norm import DiffusersStyleRMSNorm


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

        # Query/Key normalization - using diffusers-style for consistency
        self.norm_q = DiffusersStyleRMSNorm(self.head_dim, eps=1e-6)
        self.norm_k = DiffusersStyleRMSNorm(self.head_dim, eps=1e-6)
        self.norm_added_q = DiffusersStyleRMSNorm(self.head_dim, eps=1e-6)
        self.norm_added_k = DiffusersStyleRMSNorm(self.head_dim, eps=1e-6)

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

        # 1c. Concatenate results [text, image] like Flux
        joint_q = mx.concatenate([txt_q, img_q], axis=2)
        joint_k = mx.concatenate([txt_k, img_k], axis=2)
        joint_v = mx.concatenate([txt_v, img_v], axis=2)

        # 1d. Apply RoPE to concatenated Q,K
        joint_q, joint_k = QwenAttention._apply_rotary_embeddings_joint(
            joint_q=joint_q,
            joint_k=joint_k,
            txt_seq_len=txt_q.shape[2],
            image_rotary_emb=image_rotary_emb,
        )

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
    def _apply_rotary_embeddings_joint(
        joint_q: mx.array,
        joint_k: mx.array,
        txt_seq_len: int,
        image_rotary_emb: tuple[mx.array, mx.array],
    ) -> tuple[mx.array, mx.array]:
        img_rot, txt_rot = image_rotary_emb

        # Extract separate parts
        txt_q = joint_q[:, :, :txt_seq_len, :]
        img_q = joint_q[:, :, txt_seq_len:, :]
        txt_k = joint_k[:, :, :txt_seq_len, :]
        img_k = joint_k[:, :, txt_seq_len:, :]

        # Prepare cos/sin for text and image
        img_cos = img_rot[..., 0, 0].reshape(img_rot.shape[2], img_rot.shape[3])
        img_sin = img_rot[..., 1, 0].reshape(img_rot.shape[2], img_rot.shape[3])
        txt_cos = txt_rot[..., 0, 0].reshape(txt_rot.shape[2], txt_rot.shape[3])
        txt_sin = txt_rot[..., 1, 0].reshape(txt_rot.shape[2], txt_rot.shape[3])

        # Transpose [B,H,S,D] -> [B,S,H,D] for RoPE application
        img_q_bshd = mx.transpose(img_q, (0, 2, 1, 3))
        img_k_bshd = mx.transpose(img_k, (0, 2, 1, 3))
        txt_q_bshd = mx.transpose(txt_q, (0, 2, 1, 3))
        txt_k_bshd = mx.transpose(txt_k, (0, 2, 1, 3))

        # Apply RoPE
        img_q_bshd, img_k_bshd = AttentionUtils.apply_rope_bshd(img_q_bshd, img_k_bshd, img_cos, img_sin)
        txt_q_bshd, txt_k_bshd = AttentionUtils.apply_rope_bshd(txt_q_bshd, txt_k_bshd, txt_cos, txt_sin)

        # Back to [B,H,S,D]
        img_q = mx.transpose(img_q_bshd, (0, 2, 1, 3))
        img_k = mx.transpose(img_k_bshd, (0, 2, 1, 3))
        txt_q = mx.transpose(txt_q_bshd, (0, 2, 1, 3))
        txt_k = mx.transpose(txt_k_bshd, (0, 2, 1, 3))

        # Concatenate back [text, image]
        joint_q = mx.concatenate([txt_q, img_q], axis=2)
        joint_k = mx.concatenate([txt_k, img_k], axis=2)

        return joint_q, joint_k
