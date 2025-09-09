from __future__ import annotations

import mlx.core as mx
from mlx import nn

from mflux.models.qwen.model.qwen_transformer.qwen_attention import QwenAttention
from mflux.models.qwen.model.qwen_transformer.qwen_feed_forward import QwenFeedForward
from mflux.models.qwen.model.qwen_transformer.qwen_layer_norm import QwenLayerNorm


class QwenTransformerBlock(nn.Module):
    def __init__(self, dim: int = 3072, num_heads: int = 24, head_dim: int = 128):
        super().__init__()
        self.img_norm1 = QwenLayerNorm(dim=dim)
        self.txt_norm1 = QwenLayerNorm(dim=dim)
        self.attn = QwenAttention(dim=dim, num_heads=num_heads, head_dim=head_dim)
        self.img_norm2 = nn.LayerNorm(dims=dim, eps=1e-6, affine=False)
        self.txt_norm2 = nn.LayerNorm(dims=dim, eps=1e-6, affine=False)
        self.img_ff = QwenFeedForward(dim=dim)
        self.txt_ff = QwenFeedForward(dim=dim)

    def __call__(
        self,
        hidden_states: mx.array,
        encoder_hidden_states: mx.array,
        encoder_hidden_states_mask: mx.array | None,
        text_embeddings: mx.array,
        image_rotary_emb: tuple[mx.array, mx.array],
    ) -> tuple[mx.array, mx.array]:
        # 1. Compute stage 1 normalization and modulation
        img_modulated, img_gate1, img_mod2 = self.img_norm1(hidden_states, text_embeddings)
        txt_modulated, txt_gate1, txt_mod2 = self.txt_norm1(encoder_hidden_states, text_embeddings)

        # 2. Compute attention
        img_attn_output, txt_attn_output = self.attn(
            img_modulated=img_modulated,
            txt_modulated=txt_modulated,
            encoder_hidden_states_mask=encoder_hidden_states_mask,
            image_rotary_emb=image_rotary_emb,
        )

        # 3. Apply attention residual and feed forward
        hidden_states = QwenTransformerBlock._apply_residual_and_feed_forward(
            hidden_states=hidden_states,
            output=img_attn_output,
            gate_attn=img_gate1,
            mod_params=img_mod2,
            norm2=self.img_norm2,
            ff_module=self.img_ff,
        )
        encoder_hidden_states = QwenTransformerBlock._apply_residual_and_feed_forward(
            hidden_states=encoder_hidden_states,
            output=txt_attn_output,
            gate_attn=txt_gate1,
            mod_params=txt_mod2,
            norm2=self.txt_norm2,
            ff_module=self.txt_ff,
        )

        return encoder_hidden_states, hidden_states

    @staticmethod
    def _apply_residual_and_feed_forward(
        hidden_states: mx.array,
        output: mx.array,
        gate_attn: mx.array,
        mod_params: mx.array,
        norm2: nn.LayerNorm,
        ff_module: QwenFeedForward,
    ) -> mx.array:
        hidden_states = hidden_states + gate_attn[:, None, :] * output
        shift2, scale2, gate_ff = mx.split(mod_params, 3, axis=-1)
        normed = norm2(hidden_states)
        modulated = normed * (1 + scale2[:, None, :]) + shift2[:, None, :]
        ff_output = ff_module(modulated)
        return hidden_states + gate_ff[:, None, :] * ff_output.astype(hidden_states.dtype)
