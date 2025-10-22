from __future__ import annotations

import mlx.core as mx
from mlx import nn

from mflux.models.qwen.model.qwen_transformer.qwen_attention import QwenAttention
from mflux.models.qwen.model.qwen_transformer.qwen_feed_forward import QwenFeedForward


class QwenTransformerBlock(nn.Module):
    def __init__(self, dim: int = 3072, num_heads: int = 24, head_dim: int = 128):
        super().__init__()
        # Match PyTorch QwenImageTransformerBlock.__init__ exactly (lines 385-414)

        # Image processing modules
        # PyTorch: self.img_mod = nn.Sequential(nn.SiLU(), nn.Linear(dim, 6 * dim, bias=True))
        # Apply manually to match old QwenLayerNorm behavior exactly
        self.img_mod_silu = nn.SiLU()
        self.img_mod_linear = nn.Linear(dim, 6 * dim, bias=True)
        # PyTorch: self.img_norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=eps)
        self.img_norm1 = nn.LayerNorm(dims=dim, eps=1e-6, affine=False)
        self.attn = QwenAttention(dim=dim, num_heads=num_heads, head_dim=head_dim)
        # PyTorch: self.img_norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=eps)
        self.img_norm2 = nn.LayerNorm(dims=dim, eps=1e-6, affine=False)
        # PyTorch: self.img_mlp = FeedForward(dim=dim, dim_out=dim, activation_fn="gelu-approximate")
        self.img_ff = QwenFeedForward(dim=dim)

        # Text processing modules
        # PyTorch: self.txt_mod = nn.Sequential(nn.SiLU(), nn.Linear(dim, 6 * dim, bias=True))
        # Apply manually to match old QwenLayerNorm behavior exactly
        self.txt_mod_silu = nn.SiLU()
        self.txt_mod_linear = nn.Linear(dim, 6 * dim, bias=True)
        # PyTorch: self.txt_norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=eps)
        self.txt_norm1 = nn.LayerNorm(dims=dim, eps=1e-6, affine=False)
        # PyTorch: self.txt_norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=eps)
        self.txt_norm2 = nn.LayerNorm(dims=dim, eps=1e-6, affine=False)
        # PyTorch: self.txt_mlp = FeedForward(dim=dim, dim_out=dim, activation_fn="gelu-approximate")
        self.txt_ff = QwenFeedForward(dim=dim)

    def __call__(
        self,
        hidden_states: mx.array,
        encoder_hidden_states: mx.array,
        encoder_hidden_states_mask: mx.array | None,
        text_embeddings: mx.array,
        image_rotary_emb: tuple[mx.array, mx.array],
        block_idx: int | None = None,  # For debugging context
    ) -> tuple[mx.array, mx.array]:
        # Match PyTorch QwenImageTransformerBlock.forward exactly (lines 421-492)

        # Get modulation parameters for both streams
        # PyTorch: img_mod_params = self.img_mod(temb)  # [B, 6*dim]
        # Match old QwenLayerNorm: temb_silu = nn.silu(text_embeddings), mod_params = mod_linear(temb_silu)
        img_mod_params = self.img_mod_linear(self.img_mod_silu(text_embeddings))  # [B, 6*dim]
        txt_mod_params = self.txt_mod_linear(self.txt_mod_silu(text_embeddings))  # [B, 6*dim]

        # Split modulation parameters for norm1 and norm2
        # PyTorch: img_mod1, img_mod2 = img_mod_params.chunk(2, dim=-1)  # Each [B, 3*dim]
        img_mod1, img_mod2 = mx.split(img_mod_params, 2, axis=-1)  # Each [B, 3*dim]
        txt_mod1, txt_mod2 = mx.split(txt_mod_params, 2, axis=-1)  # Each [B, 3*dim]

        # Process image stream - norm1 + modulation
        # PyTorch: img_normed = self.img_norm1(hidden_states)
        img_normed = self.img_norm1(hidden_states)
        # PyTorch: img_modulated, img_gate1 = self._modulate(img_normed, img_mod1)
        img_modulated, img_gate1 = QwenTransformerBlock._modulate(img_normed, img_mod1)

        # Process text stream - norm1 + modulation
        # PyTorch: txt_normed = self.txt_norm1(encoder_hidden_states)
        txt_normed = self.txt_norm1(encoder_hidden_states)
        # PyTorch: txt_modulated, txt_gate1 = self._modulate(txt_normed, txt_mod1)
        txt_modulated, txt_gate1 = QwenTransformerBlock._modulate(txt_normed, txt_mod1)

        # Compute attention
        img_attn_output, txt_attn_output = self.attn(
            img_modulated=img_modulated,
            txt_modulated=txt_modulated,
            encoder_hidden_states_mask=encoder_hidden_states_mask,
            image_rotary_emb=image_rotary_emb,
            block_idx=block_idx,
        )

        # Apply attention gates and add residual (like in Megatron)
        # PyTorch: hidden_states = hidden_states + img_gate1 * img_attn_output
        hidden_states = hidden_states + img_gate1 * img_attn_output
        # PyTorch: encoder_hidden_states = encoder_hidden_states + txt_gate1 * txt_attn_output
        encoder_hidden_states = encoder_hidden_states + txt_gate1 * txt_attn_output

        # Process image stream - norm2 + MLP
        # PyTorch: img_normed2 = self.img_norm2(hidden_states)
        img_normed2 = self.img_norm2(hidden_states)
        # PyTorch: img_modulated2, img_gate2 = self._modulate(img_normed2, img_mod2)
        img_modulated2, img_gate2 = QwenTransformerBlock._modulate(img_normed2, img_mod2)

        # PyTorch: img_mlp_output = self.img_mlp(img_modulated2)
        img_mlp_output = self.img_ff(img_modulated2)

        # PyTorch: hidden_states = hidden_states + img_gate2 * img_mlp_output
        hidden_states = hidden_states + img_gate2 * img_mlp_output

        # Process text stream - norm2 + MLP
        # PyTorch: txt_normed2 = self.txt_norm2(encoder_hidden_states)
        txt_normed2 = self.txt_norm2(encoder_hidden_states)
        # PyTorch: txt_modulated2, txt_gate2 = self._modulate(txt_normed2, txt_mod2)
        txt_modulated2, txt_gate2 = QwenTransformerBlock._modulate(txt_normed2, txt_mod2)
        # PyTorch: txt_mlp_output = self.txt_mlp(txt_modulated2)
        txt_mlp_output = self.txt_ff(txt_modulated2)
        # PyTorch: encoder_hidden_states = encoder_hidden_states + txt_gate2 * txt_mlp_output
        encoder_hidden_states = encoder_hidden_states + txt_gate2 * txt_mlp_output

        return encoder_hidden_states, hidden_states

    @staticmethod
    def _modulate(x: mx.array, mod_params: mx.array) -> tuple[mx.array, mx.array]:
        """
        Apply modulation to input tensor (matches PyTorch _modulate method, line 416-419).

        PyTorch:
            shift, scale, gate = mod_params.chunk(3, dim=-1)
            return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1), gate.unsqueeze(1)
        """
        shift, scale, gate = mx.split(mod_params, 3, axis=-1)
        # PyTorch uses Python 1 - match exactly (MLX will handle dtype promotion)
        return x * (1 + scale[:, None, :]) + shift[:, None, :], gate[:, None, :]
