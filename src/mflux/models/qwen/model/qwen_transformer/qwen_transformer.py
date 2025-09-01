from __future__ import annotations

import mlx.core as mx
import numpy as np
from mlx import nn

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

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_layers = num_layers
        self.attention_head_dim = attention_head_dim
        self.num_attention_heads = num_attention_heads
        self.joint_attention_dim = joint_attention_dim
        self.patch_size = patch_size

        inner_dim = num_attention_heads * attention_head_dim
        self.inner_dim = inner_dim

        # Input projections
        self.img_in = nn.Linear(in_channels, inner_dim)
        self.txt_norm = nn.RMSNorm(joint_attention_dim, eps=1e-6)
        self.txt_in = nn.Linear(joint_attention_dim, inner_dim)

        # Time/Text embedding
        self.time_text_embed = QwenTimeTextEmbed(timestep_proj_dim=256, inner_dim=inner_dim)

        # RoPE helper
        self.pos_embed = QwenEmbedRopeMLX(theta=10000, axes_dim=[16, 56, 56], scale_rope=True)

        # Blocks
        self.transformer_blocks = [
            QwenTransformerBlock(dim=inner_dim, num_heads=num_attention_heads, head_dim=attention_head_dim)
            for i in range(num_layers)
        ]

        # Output head
        self.norm_out = AdaLayerNormContinuous(inner_dim, inner_dim)
        self.proj_out = nn.Linear(inner_dim, patch_size * patch_size * out_channels)

    def __call__(
        self,
        t: int,
        config,
        hidden_states: mx.array,
        encoder_hidden_states: mx.array,
        encoder_hidden_states_mask: mx.array,
    ) -> mx.array:
        # Compute internal flux_transformer details
        latent_height = config.height // 16
        latent_width = config.width // 16
        img_shapes = [(1, latent_height, latent_width)]
        txt_seq_lens = [
            int(mx.sum(encoder_hidden_states_mask[i]).item()) for i in range(encoder_hidden_states_mask.shape[0])
        ]

        # Resolve timestep from t and config (like Flux)
        timestep_value = config.get_qwen_timestep(t)
        batch = hidden_states.shape[0]
        timestep = mx.array(np.full((batch,), timestep_value, dtype=np.float32))

        hs = self.img_in(hidden_states)
        txt = self.txt_in(self.txt_norm(encoder_hidden_states))
        temb = self.time_text_embed(timestep, hs)
        img_rot, txt_rot = self.pos_embed(video_fhw=img_shapes, txt_seq_lens=txt_seq_lens)

        # Blocks
        for idx, block in enumerate(self.transformer_blocks):
            txt, hs = block(
                hidden_states=hs,
                encoder_hidden_states=txt,
                encoder_hidden_states_mask=encoder_hidden_states_mask,
                temb=temb,
                image_rotary_emb=(img_rot, txt_rot),
            )

        # Output head (use only image stream)
        hs = self.norm_out(hs, temb)
        out = self.proj_out(hs)
        return out
