from __future__ import annotations

import mlx.core as mx
import numpy as np
from mlx import nn

from mflux.models.common.config.config import Config
from mflux.models.flux.model.flux_transformer.ada_layer_norm_continuous import AdaLayerNormContinuous
from mflux.models.qwen.model.qwen_transformer.qwen_time_text_embed import QwenTimeTextEmbed
from mflux.models.qwen.model.qwen_transformer.qwen_transformer_block import QwenTransformerBlock
from mflux.models.qwen.model.qwen_transformer.qwen_transformer_rms_norm import QwenTransformerRMSNorm
from mflux.models.qwen_layered.model.qwen_layered_transformer.qwen_layered_rope import QwenLayeredRoPE


class QwenLayeredTransformer(nn.Module):
    """
    VLD-MMDiT (Variable Layers Decomposition MMDiT) for Qwen-Image-Layered.
    
    Key differences from base transformer:
    1. Uses Layer3D RoPE instead of standard 2D RoPE
    2. Accepts condition image latent (z_I) alongside noisy layers (x_t)
    3. Concatenates z_I and x_t along sequence dimension for joint attention
    4. Handles variable number of output layers N
    """

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
        self.patch_size = patch_size
        self.out_channels = out_channels
        
        # Input projections
        self.img_in = nn.Linear(in_channels, self.inner_dim)
        self.cond_img_in = nn.Linear(in_channels, self.inner_dim)  # For condition image
        
        # Text processing
        self.txt_norm = QwenTransformerRMSNorm(joint_attention_dim, eps=1e-6)
        self.txt_in = nn.Linear(joint_attention_dim, self.inner_dim)
        
        # Time embedding
        self.time_text_embed = QwenTimeTextEmbed(timestep_proj_dim=256, inner_dim=self.inner_dim)
        
        # Layer3D RoPE instead of standard RoPE
        self.pos_embed = QwenLayeredRoPE(theta=10000, axes_dim=[16, 56, 56], scale_rope=True)
        
        # Transformer blocks (same as base)
        self.transformer_blocks = [
            QwenTransformerBlock(dim=self.inner_dim, num_heads=num_attention_heads, head_dim=attention_head_dim)
            for _ in range(num_layers)
        ]
        
        # Output
        self.norm_out = AdaLayerNormContinuous(self.inner_dim, self.inner_dim)
        self.proj_out = nn.Linear(self.inner_dim, patch_size * patch_size * out_channels)

    def __call__(
        self,
        t: int,
        config: Config,
        hidden_states: mx.array,
        cond_image_hidden_states: mx.array,
        encoder_hidden_states: mx.array,
        encoder_hidden_states_mask: mx.array,
        num_output_layers: int,
    ) -> mx.array:
        """
        Forward pass for layered decomposition.
        
        Args:
            t: Current timestep
            config: Generation config
            hidden_states: Noisy layer latents [B, N*seq_len, C]
            cond_image_hidden_states: Condition image latents [B, seq_len, C]
            encoder_hidden_states: Text embeddings [B, txt_len, C]
            encoder_hidden_states_mask: Text attention mask
            num_output_layers: Number of output layers N
            
        Returns:
            Predicted noise [B, N*seq_len, out_C]
        """
        batch_size = hidden_states.shape[0]
        
        # Project inputs
        hidden_states = self.img_in(hidden_states)
        cond_image_hidden_states = self.cond_img_in(cond_image_hidden_states)
        
        # Concatenate condition image and noisy layers along sequence dimension
        # [B, seq + N*seq, C]
        combined_hidden_states = mx.concatenate([cond_image_hidden_states, hidden_states], axis=1)
        
        # Compute timestep
        timestep = self._compute_timestep(t, config)
        timestep = mx.broadcast_to(timestep, (batch_size,)).astype(combined_hidden_states.dtype)
        
        # Process text
        encoder_hidden_states = self.txt_norm(encoder_hidden_states)
        encoder_hidden_states = self.txt_in(encoder_hidden_states)
        
        # Time embedding
        text_embeddings = self.time_text_embed(timestep, combined_hidden_states)
        
        # Compute Layer3D RoPE
        latent_height = config.height // 16
        latent_width = config.width // 16
        txt_seq_lens = [int(mx.sum(encoder_hidden_states_mask[i]).item()) for i in range(encoder_hidden_states_mask.shape[0])]
        
        image_rotary_emb, txt_rotary_emb = self.pos_embed(
            num_layers=num_output_layers,
            height=latent_height,
            width=latent_width,
            txt_seq_lens=txt_seq_lens,
            include_cond_image=True,
        )
        
        # Apply transformer blocks
        for idx, block in enumerate(self.transformer_blocks):
            encoder_hidden_states, combined_hidden_states = block(
                hidden_states=combined_hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                encoder_hidden_states_mask=encoder_hidden_states_mask,
                text_embeddings=text_embeddings,
                image_rotary_emb=(image_rotary_emb, txt_rotary_emb),
                block_idx=idx,
            )
        
        # Split out the noisy layers (remove condition image)
        cond_seq_len = cond_image_hidden_states.shape[1]
        hidden_states = combined_hidden_states[:, cond_seq_len:, :]
        
        # Output projection
        hidden_states = self.norm_out(hidden_states, text_embeddings)
        hidden_states = self.proj_out(hidden_states)
        
        return hidden_states

    @staticmethod
    def _compute_timestep(t: int | float, config: Config) -> mx.array:
        """Compute timestep value from step index."""
        if isinstance(t, int):
            if t < len(config.scheduler.sigmas):
                time_step = config.scheduler.sigmas[t]
            else:
                timestep_idx = None
                for idx, ts in enumerate(config.scheduler.timesteps):
                    if abs(int(ts.item()) - t) < 1:
                        timestep_idx = idx
                        break
                if timestep_idx is None:
                    time_step = t / 1000.0
                else:
                    time_step = config.scheduler.sigmas[timestep_idx]
        else:
            time_step = t

        timestep = mx.array(np.full((1,), time_step, dtype=np.float32))
        return timestep
