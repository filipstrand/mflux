from __future__ import annotations

import mlx.core as mx
import numpy as np
from mlx import nn

from mflux.config.runtime_config import RuntimeConfig
from mflux.models.flux.model.flux_transformer.ada_layer_norm_continuous import AdaLayerNormContinuous
from mflux.models.qwen.model.qwen_transformer.qwen_rope import QwenEmbedRopeMLX
from mflux.models.qwen.model.qwen_transformer.qwen_time_text_embed import QwenTimeTextEmbed
from mflux.models.qwen.model.qwen_transformer.qwen_transformer_block import QwenTransformerBlock


class QwenTransformerRMSNorm(nn.Module):
    """
    RMSNorm that matches PyTorch's RMSNorm implementation exactly (diffusers/models/normalization.py:511-568).
    Only variance calculation uses float32, main computation stays in original dtype.
    """

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        # PyTorch: elementwise_affine defaults to True, so we create weight parameter
        # PyTorch: self.weight = nn.Parameter(torch.ones(dim))
        self.weight = mx.ones((dim,))
        self.eps = eps

    def __call__(self, hidden_states: mx.array) -> mx.array:
        """
        Matches PyTorch RMSNorm.forward exactly (lines 554-566).

        PyTorch:
            input_dtype = hidden_states.dtype
            variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
            hidden_states = hidden_states * torch.rsqrt(variance + self.eps)
            if self.weight is not None:
                if self.weight.dtype in [torch.float16, torch.bfloat16]:
                    hidden_states = hidden_states.to(self.weight.dtype)
                hidden_states = hidden_states * self.weight
        """
        # PyTorch: input_dtype = hidden_states.dtype
        input_dtype = hidden_states.dtype

        # PyTorch: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        # Only variance calculation uses float32
        variance = mx.power(hidden_states.astype(mx.float32), 2).mean(axis=-1, keepdims=True)

        # PyTorch: hidden_states = hidden_states * torch.rsqrt(variance + self.eps)
        # Main computation stays in original dtype
        hidden_states = hidden_states * mx.rsqrt(variance + self.eps)

        # PyTorch: if self.weight is not None:
        if self.weight is not None:
            # PyTorch: if self.weight.dtype in [torch.float16, torch.bfloat16]:
            #          hidden_states = hidden_states.to(self.weight.dtype)
            # Match PyTorch: convert to weight dtype if half-precision
            if self.weight.dtype in [mx.bfloat16, mx.float16]:
                hidden_states = hidden_states.astype(self.weight.dtype)
            # PyTorch: hidden_states = hidden_states * self.weight
            hidden_states = hidden_states * self.weight
            # Match PyTorch: convert back to input dtype if needed
            if hidden_states.dtype != input_dtype:
                hidden_states = hidden_states.astype(input_dtype)

        return hidden_states


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
        # PyTorch: self.txt_norm = RMSNorm(joint_attention_dim, eps=1e-6, elementwise_affine=False)
        # Use custom RMSNorm to match PyTorch's dtype handling exactly
        self.txt_norm = QwenTransformerRMSNorm(joint_attention_dim, eps=1e-6)
        self.txt_in = nn.Linear(joint_attention_dim, self.inner_dim)
        self.time_text_embed = QwenTimeTextEmbed(timestep_proj_dim=256, inner_dim=self.inner_dim)
        self.pos_embed = QwenEmbedRopeMLX(theta=10000, axes_dim=[16, 56, 56], scale_rope=True)
        self.transformer_blocks = []
        for i in range(num_layers):
            block = QwenTransformerBlock(
                dim=self.inner_dim,
                num_heads=num_attention_heads,
                head_dim=attention_head_dim,
            )
            self.transformer_blocks.append(block)
            setattr(self, f"transformer_block_{i}", block)
        self.norm_out = AdaLayerNormContinuous(self.inner_dim, self.inner_dim)
        self.proj_out = nn.Linear(self.inner_dim, patch_size * patch_size * out_channels)

    def update(self, values, strict: bool = True):
        values = dict(values)
        transformer_blocks_state = values.pop("transformer_blocks", None)
        super().update(values, strict=strict)
        if transformer_blocks_state is not None:
            for block, block_state in zip(self.transformer_blocks, transformer_blocks_state):
                block.update(block_state, strict=strict)
        return self

    def __call__(
        self,
        t: int,
        config: RuntimeConfig,
        hidden_states: mx.array,
        encoder_hidden_states: mx.array,
        encoder_hidden_states_mask: mx.array,
        qwen_image_ids: mx.array | None = None,  # Optional: for Edit model
        cond_image_grid: tuple[int, int, int] | None = None,  # Optional: for Edit model
    ) -> mx.array:
        # Match PyTorch QwenImageTransformer.forward exactly (lines 637-695)

        # 1. Image input projection
        # PyTorch: hidden_states = self.img_in(hidden_states)
        hidden_states = self.img_in(hidden_states)

        # 2. Convert timestep to hidden_states dtype BEFORE time_text_embed
        # PyTorch: timestep = timestep.to(hidden_states.dtype)
        batch_size = hidden_states.shape[0]
        timestep = QwenTransformer._compute_timestep(t, config)
        timestep = mx.broadcast_to(timestep, (batch_size,)).astype(hidden_states.dtype)

        # 3. Text normalization and input projection
        # PyTorch: encoder_hidden_states = self.txt_norm(encoder_hidden_states)
        # PyTorch: encoder_hidden_states = self.txt_in(encoder_hidden_states)
        encoder_hidden_states = self.txt_norm(encoder_hidden_states)
        encoder_hidden_states = self.txt_in(encoder_hidden_states)

        # 4. Compute text embeddings (time + text conditioning)
        # PyTorch: temb = self.time_text_embed(timestep, hidden_states)
        text_embeddings = self.time_text_embed(timestep, hidden_states)

        # 5. Compute rotary embeddings
        # PyTorch: image_rotary_emb = self.pos_embed(img_shapes, txt_seq_lens, device=hidden_states.device)
        image_rotary_embeddings = QwenTransformer._compute_rotary_embeddings(
            encoder_hidden_states_mask=encoder_hidden_states_mask,
            pos_embed=self.pos_embed,
            config=config,
            cond_image_grid=cond_image_grid,
        )

        # 6. Run the transformer blocks
        # PyTorch: for index_block, block in enumerate(self.transformer_blocks): ...
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

        # 7. Apply output normalization and projection
        # PyTorch: hidden_states = self.norm_out(hidden_states, temb)
        # PyTorch: output = self.proj_out(hidden_states)
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
            block_idx=idx,  # Pass block index for debugging context
        )

    @staticmethod
    def _compute_timestep(
        t: int | float,
        config: RuntimeConfig,
    ) -> mx.array:
        """
        Compute timestep tensor from step index or value.
        Matches PyTorch's timestep handling (before conversion to hidden_states.dtype).

        Returns:
            timestep array [batch_size] - will be converted to hidden_states.dtype in __call__
        """
        # Handle two cases:
        # 1. txt2img: passes indices (0, 1, 2...) which are small integers
        # 2. edit: passes actual timestep values (1000, 967, 932...) which are large integers
        # 3. If t is a float, it's already the sigma value
        if isinstance(t, int):
            # Check if t is a small index or a large timestep value
            if t < len(config.scheduler.sigmas):
                # Case 1: Small value, likely an index (txt2img)
                timestep_idx = t
                time_step = config.scheduler.sigmas[timestep_idx]
            else:
                # Case 2: Large value, likely an actual timestep (edit model)
                # Find which sigma corresponds to this timestep value
                timestep_idx = None
                for idx, ts in enumerate(config.scheduler.timesteps):
                    if abs(int(ts.item()) - t) < 1:  # Allow small tolerance for rounding
                        timestep_idx = idx
                        break
                if timestep_idx is None:
                    # Fallback: use the timestep value directly as sigma (might be pre-normalized)
                    time_step = t / 1000.0  # Normalize to [0, 1] range
                else:
                    time_step = config.scheduler.sigmas[timestep_idx]
        else:
            timestep_idx = None
            time_step = t

        # Create timestep tensor (will be converted to hidden_states.dtype in __call__)
        # PyTorch receives timestep as LongTensor, then converts to hidden_states.dtype
        # We create as float32 first, then convert in __call__
        timestep = mx.array(np.full((1,), time_step, dtype=np.float32))
        return timestep

    @staticmethod
    def _compute_rotary_embeddings(
        encoder_hidden_states_mask: mx.array,
        pos_embed: QwenEmbedRopeMLX,
        config: RuntimeConfig,
        cond_image_grid: tuple[int, int, int] | list[tuple[int, int, int]] | None = None,
    ) -> tuple[mx.array, mx.array]:
        # CRITICAL: Latents are packed (2x2), so patch counts are already divided by 4
        # Generation latents: height//16, width//16 (already packed)
        # Conditioning latents: cond_h_patches, cond_w_patches (already packed)
        # RoPE frequencies need to match the actual sequence length of packed latents
        latent_height = config.height // 16
        latent_width = config.width // 16

        # For txt2img (simple case), just use generation shape
        # For Edit model with conditioning image(s), pass list of shapes
        if cond_image_grid is None:
            img_shapes = [(1, latent_height, latent_width)]
        else:
            # Edit model needs both generation and conditioning shapes
            # Pass both to generate RoPE embeddings for concatenated latents
            # If cond_image_grid is a list, it means multiple conditioning images
            # NOTE: cond_image_grid already contains packed patch counts (from EditUtil)
            if isinstance(cond_image_grid, list):
                # Multiple conditioning images: [gen_shape, cond1, cond2, ...]
                img_shapes = [(1, latent_height, latent_width)] + cond_image_grid
            else:
                # Single conditioning image: [gen_shape, cond_shape]
                img_shapes = [(1, latent_height, latent_width), cond_image_grid]

        txt_seq_lens = [int(mx.sum(encoder_hidden_states_mask[i]).item()) for i in range(encoder_hidden_states_mask.shape[0])]  # fmt: off
        img_rotary_emb, txt_rotary_emb = pos_embed(video_fhw=img_shapes, txt_seq_lens=txt_seq_lens)
        return img_rotary_emb, txt_rotary_emb
