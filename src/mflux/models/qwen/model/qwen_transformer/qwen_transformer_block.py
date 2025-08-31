from __future__ import annotations

import mlx.core as mx
from mlx import nn

from mflux.models.flux.model.flux_transformer.common.attention_utils import AttentionUtils


class QwenTransformerBlock(nn.Module):
    def __init__(self, dim: int = 3072, num_heads: int = 24, head_dim: int = 128):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.debug: bool = False

        # Modulation (SiLU + Linear to 6*dim for shift/scale/gate twice)
        self.img_mod_linear = nn.Linear(dim, 6 * dim)
        self.txt_mod_linear = nn.Linear(dim, 6 * dim)

        # Norms (elementwise_affine False)
        self.img_norm1 = nn.LayerNorm(dims=dim, eps=1e-6, affine=False)
        self.txt_norm1 = nn.LayerNorm(dims=dim, eps=1e-6, affine=False)

        # Attention projections and norms
        self.to_q = nn.Linear(dim, dim)
        self.to_k = nn.Linear(dim, dim)
        self.to_v = nn.Linear(dim, dim)

        self.add_q_proj = nn.Linear(dim, dim)
        self.add_k_proj = nn.Linear(dim, dim)
        self.add_v_proj = nn.Linear(dim, dim)

        self.norm_q = nn.RMSNorm(self.head_dim, eps=1e-6)
        self.norm_k = nn.RMSNorm(self.head_dim, eps=1e-6)
        self.norm_added_q = nn.RMSNorm(self.head_dim, eps=1e-6)
        self.norm_added_k = nn.RMSNorm(self.head_dim, eps=1e-6)

        self.attn_to_out = [nn.Linear(dim, dim)]
        self.to_add_out = nn.Linear(dim, dim)

        # Second-stage norms and MLPs
        self.img_norm2 = nn.LayerNorm(dims=dim, eps=1e-6, affine=False)
        self.txt_norm2 = nn.LayerNorm(dims=dim, eps=1e-6, affine=False)

        self.img_mlp_in = nn.Linear(dim, 4 * dim)
        self.img_mlp_out = nn.Linear(4 * dim, dim)

        self.txt_mlp_in = nn.Linear(dim, 4 * dim)
        self.txt_mlp_out = nn.Linear(4 * dim, dim)

    def __call__(
        self,
        hidden_states: mx.array,
        encoder_hidden_states: mx.array,
        encoder_hidden_states_mask: mx.array | None,
        temb: mx.array,
        image_rotary_emb: tuple[mx.array, mx.array],
    ) -> tuple[mx.array, mx.array]:
        # Modulation params
        img_mod_params = self.img_mod_linear(nn.silu(temb))
        txt_mod_params = self.txt_mod_linear(nn.silu(temb))

        img_mod1, img_mod2 = mx.split(img_mod_params, 2, axis=-1)
        txt_mod1, txt_mod2 = mx.split(txt_mod_params, 2, axis=-1)

        # Norm + modulate (stage 1)
        img_normed = self.img_norm1(hidden_states)
        img_modulated, img_gate1 = self._modulate(img_normed, img_mod1)

        txt_normed = self.txt_norm1(encoder_hidden_states)
        txt_modulated, txt_gate1 = self._modulate(txt_normed, txt_mod1)

        # Attention: compute QKV for image and text streams
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

        # Apply RoPE separately in [B,H,S,D] layout using real mixing with per-stream lengths
        img_rot, txt_rot = image_rotary_emb
        img_cos = img_rot[..., 0, 0].reshape(img_rot.shape[2], img_rot.shape[3])
        img_sin = img_rot[..., 1, 0].reshape(img_rot.shape[2], img_rot.shape[3])
        txt_cos = txt_rot[..., 0, 0].reshape(txt_rot.shape[2], txt_rot.shape[3])
        txt_sin = txt_rot[..., 1, 0].reshape(txt_rot.shape[2], txt_rot.shape[3])

        # Transpose [B,H,S,D] -> [B,S,H,D] for mixing convenience
        img_q_bshd = mx.transpose(img_q, (0, 2, 1, 3))
        img_k_bshd = mx.transpose(img_k, (0, 2, 1, 3))
        txt_q_bshd = mx.transpose(txt_q, (0, 2, 1, 3))
        txt_k_bshd = mx.transpose(txt_k, (0, 2, 1, 3))

        img_q_bshd, img_k_bshd = AttentionUtils.apply_rope_bshd(img_q_bshd, img_k_bshd, img_cos, img_sin)
        txt_q_bshd, txt_k_bshd = AttentionUtils.apply_rope_bshd(txt_q_bshd, txt_k_bshd, txt_cos, txt_sin)

        # Back to [B,H,S,D]
        img_q = mx.transpose(img_q_bshd, (0, 2, 1, 3))
        img_k = mx.transpose(img_k_bshd, (0, 2, 1, 3))
        txt_q = mx.transpose(txt_q_bshd, (0, 2, 1, 3))
        txt_k = mx.transpose(txt_k_bshd, (0, 2, 1, 3))

        # Concatenate [text, image]
        joint_q = mx.concatenate([txt_q, img_q], axis=2)
        joint_k = mx.concatenate([txt_k, img_k], axis=2)
        joint_v = mx.concatenate([txt_v, img_v], axis=2)

        key_padding_mask = None
        if encoder_hidden_states_mask is not None:
            bsz = hidden_states.shape[0]
            s_img = hidden_states.shape[1]
            ones_img = mx.ones((bsz, s_img), dtype=mx.float32)
            key_padding_mask = mx.concatenate([encoder_hidden_states_mask.astype(mx.float32), ones_img], axis=1)

        # Compute attention explicitly (optionally with key padding mask)
        joint_hs = AttentionUtils.compute_attention_explicit(joint_q, joint_k, joint_v, key_padding_mask)
        joint_hs = joint_hs.astype(joint_q.dtype)

        # Split back
        seq_txt = encoder_hidden_states.shape[1]
        txt_attn_output = joint_hs[:, :seq_txt, :]
        img_attn_output = joint_hs[:, seq_txt:, :]

        # Output projections (compute in float32 like working version, then cast back)
        img_dtype = img_attn_output.dtype
        txt_dtype = txt_attn_output.dtype
        img_attn_output = self.attn_to_out[0](img_attn_output.astype(mx.float32)).astype(img_dtype)
        txt_attn_output = self.to_add_out(txt_attn_output.astype(mx.float32)).astype(txt_dtype)

        # Apply gates and residual (stage 1)
        hidden_states = hidden_states + img_gate1[:, None, :] * img_attn_output
        encoder_hidden_states = encoder_hidden_states + txt_gate1[:, None, :] * txt_attn_output

        # Stage 2: norm + MLP for both streams
        img_normed2 = self.img_norm2(hidden_states)
        img_modulated2, img_gate2 = self._modulate(img_normed2, img_mod2)
        img_mlp_hidden = self.img_mlp_in(img_modulated2.astype(mx.float32))
        img_mlp_hidden = nn.gelu_approx(img_mlp_hidden)
        img_mlp_out = self.img_mlp_out(img_mlp_hidden).astype(hidden_states.dtype)
        hidden_states = hidden_states + img_gate2[:, None, :] * img_mlp_out

        txt_normed2 = self.txt_norm2(encoder_hidden_states)
        txt_modulated2, txt_gate2 = self._modulate(txt_normed2, txt_mod2)
        txt_mlp_hidden = self.txt_mlp_in(txt_modulated2.astype(mx.float32))
        txt_mlp_hidden = nn.gelu_approx(txt_mlp_hidden)
        txt_mlp_out = self.txt_mlp_out(txt_mlp_hidden).astype(encoder_hidden_states.dtype)
        encoder_hidden_states = encoder_hidden_states + txt_gate2[:, None, :] * txt_mlp_out

        return encoder_hidden_states, hidden_states

    @staticmethod
    def _modulate(x: mx.array, mod_params: mx.array) -> tuple[mx.array, mx.array]:
        shift, scale, gate = mx.split(mod_params, 3, axis=-1)
        return x * (1 + scale[:, None, :]) + shift[:, None, :], gate
