from __future__ import annotations

import mlx.core as mx
import numpy as np
from mlx import nn

from mflux.config.runtime_config import RuntimeConfig
from mflux.models.flux.model.flux_transformer.ada_layer_norm_continuous import AdaLayerNormContinuous
from mflux.models.qwen.model.qwen_transformer.qwen_rope import QwenEmbedRopeMLX
from mflux.models.qwen.model.qwen_transformer.qwen_time_text_embed import QwenTimeTextEmbed
from mflux.models.qwen.model.qwen_transformer.qwen_transformer_block import QwenTransformerBlock


class QwenTransformer(nn.Module):
    def __init__(
        self,
        in_channels: int = 64,
        out_channels: int = 16,
        num_layers: int = 60,
        attention_head_dim: int = 128,
        num_attention_heads: int = 24,
        joint_attention_dim: int = 3584,
        patch_size: int = 2,
    ) -> None:
        super().__init__()
        self.inner_dim = num_attention_heads * attention_head_dim
        self.img_in = nn.Linear(in_channels, self.inner_dim)
        self.txt_norm = nn.RMSNorm(joint_attention_dim, eps=1e-6)
        self.txt_in = nn.Linear(joint_attention_dim, self.inner_dim)
        self.time_text_embed = QwenTimeTextEmbed(timestep_proj_dim=256, inner_dim=self.inner_dim)
        self.pos_embed = QwenEmbedRopeMLX(theta=10000, axes_dim=[16, 56, 56], scale_rope=True)
        self.transformer_blocks = [QwenTransformerBlock(dim=self.inner_dim, num_heads=num_attention_heads, head_dim=attention_head_dim) for i in range(num_layers)]  # fmt: off
        self.norm_out = AdaLayerNormContinuous(self.inner_dim, self.inner_dim)
        self.proj_out = nn.Linear(self.inner_dim, patch_size * patch_size * out_channels)

    def __call__(
        self,
        t: int,
        config: RuntimeConfig,
        hidden_states: mx.array,
        encoder_hidden_states: mx.array,
        encoder_hidden_states_mask: mx.array,
    ) -> mx.array:
        # 1. Create embeddings
        hidden_states = self.img_in(hidden_states)
        encoder_hidden_states = self.txt_in(self.txt_norm(encoder_hidden_states))
        text_embeddings = QwenTransformer._compute_text_embeddings(t, hidden_states, self.time_text_embed, config)
        image_rotary_embeddings = QwenTransformer._compute_rotary_embeddings(encoder_hidden_states_mask, self.pos_embed, config)  # fmt: off

        # 2. Run the transformer blocks
        for idx, block in enumerate(self.transformer_blocks):
            encoder_hidden_states, hidden_states = QwenTransformer._apply_transformer_block(
                idx=idx,
                block=block,
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                encoder_hidden_states_mask=encoder_hidden_states_mask,
                text_embeddings=text_embeddings,
                image_rotary_embeddings=image_rotary_embeddings,
            )

        # 3. Apply output normalization and projection
        hidden_states = self.norm_out(hidden_states, text_embeddings)
        hidden_states = self.proj_out(hidden_states)
        return hidden_states

    @staticmethod
    def _apply_transformer_block(
        idx: int,
        block: QwenTransformerBlock,
        hidden_states: mx.array,
        encoder_hidden_states: mx.array,
        encoder_hidden_states_mask: mx.array,
        text_embeddings: mx.array,
        image_rotary_embeddings: tuple[mx.array, mx.array],
    ) -> tuple[mx.array, mx.array]:
        return block(
            hidden_states=hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            encoder_hidden_states_mask=encoder_hidden_states_mask,
            text_embeddings=text_embeddings,
            image_rotary_emb=image_rotary_embeddings,
        )

    @staticmethod
    def _compute_text_embeddings(
        t: int,
        hidden_states: mx.array,
        time_text_embed: QwenTimeTextEmbed,
        config: RuntimeConfig,
    ) -> mx.array:
        time_step = config.scheduler.sigmas[t]
        batch = hidden_states.shape[0]
        timestep = mx.array(np.full((batch,), time_step, dtype=np.float32))
        text_embeddings = time_text_embed(timestep, hidden_states)
        return text_embeddings

    @staticmethod
    def _compute_rotary_embeddings(
        encoder_hidden_states_mask: mx.array,
        pos_embed: QwenEmbedRopeMLX,
        config: RuntimeConfig,
    ) -> tuple[mx.array, mx.array]:
        latent_height = config.height // 16
        latent_width = config.width // 16
        img_shapes = [(1, latent_height, latent_width)]
        txt_seq_lens = [int(mx.sum(encoder_hidden_states_mask[i]).item()) for i in range(encoder_hidden_states_mask.shape[0])]  # fmt: off
        img_rotary_emb, txt_rotary_emb = pos_embed(video_fhw=img_shapes, txt_seq_lens=txt_seq_lens)
        return img_rotary_emb, txt_rotary_emb
