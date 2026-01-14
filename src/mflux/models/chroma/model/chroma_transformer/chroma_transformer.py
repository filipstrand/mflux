"""
Chroma Transformer.

The main transformer model for Chroma, which differs from FLUX in:
1. Uses DistilledGuidanceLayer instead of TimeTextEmbed
2. Pre-computes all 344 modulations upfront
3. Distributes modulations to blocks based on index scheme
4. No norm1.linear weights in blocks (modulations pre-computed)

Modulation distribution:
- Indices 0-113: Single blocks (38 × 3 mods)
- Indices 114-227: Joint blocks image mods (19 × 6 mods)
- Indices 228-341: Joint blocks text mods (19 × 6 mods)
- Indices 342-343: Final norm mods (2)

IMPORTANT: Processing order in Chroma:
1. Joint blocks are processed FIRST (using mods from indices 114-341)
2. Single blocks are processed SECOND (using mods from indices 0-113)
3. Final norm is applied last (using mods from indices 342-343)
"""

import math

import mlx.core as mx
from mlx import nn

from mflux.models.chroma.model.chroma_transformer.chroma_ada_layer_norm import ChromaAdaLayerNormContinuousPruned
from mflux.models.chroma.model.chroma_transformer.chroma_joint_transformer_block import ChromaJointTransformerBlock
from mflux.models.chroma.model.chroma_transformer.chroma_single_transformer_block import ChromaSingleTransformerBlock
from mflux.models.chroma.model.chroma_transformer.distilled_guidance_layer import DistilledGuidanceLayer
from mflux.models.common.config.config import Config
from mflux.models.common.config.model_config import ModelConfig
from mflux.models.flux.model.flux_transformer.embed_nd import EmbedND


class ChromaTransformer(nn.Module):
    """
    Chroma Transformer model.

    Key architectural differences from FLUX Transformer:
    1. DistilledGuidanceLayer replaces TimeTextEmbed
    2. All modulations computed upfront and distributed to blocks
    3. No pooled_prompt_embeds parameter (no CLIP encoder)
    4. Blocks don't have norm linear layers
    """

    def __init__(
        self,
        model_config: ModelConfig,
        num_transformer_blocks: int = 19,
        num_single_transformer_blocks: int = 38,
        inner_dim: int = 3072,
        approximator_num_channels: int = 64,
        approximator_hidden_dim: int = 5120,
        approximator_layers: int = 5,
    ):
        super().__init__()
        self.inner_dim = inner_dim
        self.num_transformer_blocks = num_transformer_blocks
        self.num_single_transformer_blocks = num_single_transformer_blocks

        # Position embedding (same as FLUX)
        self.pos_embed = EmbedND()

        # Input embedders (same as FLUX)
        self.x_embedder = nn.Linear(model_config.x_embedder_input_dim(), inner_dim)
        self.context_embedder = nn.Linear(4096, inner_dim)

        # DistilledGuidanceLayer (replaces TimeTextEmbed)
        # Output dimension = 3 * 38 + 2 * 6 * 19 + 2 = 344 modulations
        out_dim = 3 * num_single_transformer_blocks + 2 * 6 * num_transformer_blocks + 2
        self.distilled_guidance_layer = DistilledGuidanceLayer(
            num_channels=approximator_num_channels,
            out_dim=out_dim,
            inner_dim=inner_dim,
            hidden_dim=approximator_hidden_dim,
            n_layers=approximator_layers,
        )

        # Transformer blocks
        self.transformer_blocks = [ChromaJointTransformerBlock(i, dim=inner_dim) for i in range(num_transformer_blocks)]
        self.single_transformer_blocks = [
            ChromaSingleTransformerBlock(i, dim=inner_dim) for i in range(num_single_transformer_blocks)
        ]

        # Output layers (note: norm_out has no linear, unlike FLUX)
        self.norm_out = ChromaAdaLayerNormContinuousPruned(dim=inner_dim)
        self.proj_out = nn.Linear(inner_dim, 64)

    def __call__(
        self,
        t: int,
        config: Config,
        hidden_states: mx.array,
        prompt_embeds: mx.array,
        controlnet_block_samples: list[mx.array] | None = None,
        controlnet_single_block_samples: list[mx.array] | None = None,
    ) -> mx.array:
        """
        Forward pass of Chroma Transformer.

        Args:
            t: Current timestep index
            config: Configuration object with scheduler and model info
            hidden_states: Latent image tensor [batch, seq_len, channels]
            prompt_embeds: T5 text embeddings [batch, text_seq_len, 4096]
            controlnet_block_samples: Optional controlnet residuals for joint blocks
            controlnet_single_block_samples: Optional controlnet residuals for single blocks

        Returns:
            Predicted noise tensor
        """
        # 1. Embed inputs
        hidden_states = self.x_embedder(hidden_states)
        encoder_hidden_states = self.context_embedder(prompt_embeds)

        # 2. Compute all modulations upfront
        timestep = config.scheduler.sigmas[t].astype(config.precision)
        pooled_temb = self.distilled_guidance_layer(timestep)

        # 3. Compute rotary embeddings
        image_rotary_embeddings = self._compute_rotary_embeddings(prompt_embeds, config)

        # 4. Run joint transformer blocks (processed FIRST)
        for idx, block in enumerate(self.transformer_blocks):
            # Get modulations for this joint block
            # Image mods: indices 114 + 6*idx to 114 + 6*idx + 6
            # Text mods: indices 228 + 6*idx to 228 + 6*idx + 6
            image_modulations, text_modulations = self._get_joint_block_modulations(pooled_temb, idx)

            encoder_hidden_states, hidden_states = block(
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                image_modulations=image_modulations,
                text_modulations=text_modulations,
                rotary_embeddings=image_rotary_embeddings,
            )

            # Apply controlnet residual if provided
            if controlnet_block_samples is not None:
                sample = self._get_controlnet_sample(idx, self.transformer_blocks, controlnet_block_samples)
                if sample is not None:
                    hidden_states = hidden_states + sample

        # 5. Concatenate hidden states
        hidden_states = mx.concatenate([encoder_hidden_states, hidden_states], axis=1)

        # 6. Run single transformer blocks (processed SECOND)
        for idx, block in enumerate(self.single_transformer_blocks):
            # Get modulations for this single block
            # Single mods: indices 3*idx to 3*idx + 3
            modulations = self._get_single_block_modulations(pooled_temb, idx)

            hidden_states = block(
                hidden_states=hidden_states,
                modulations=modulations,
                rotary_embeddings=image_rotary_embeddings,
            )

            # Apply controlnet residual if provided
            if controlnet_single_block_samples is not None:
                sample = self._get_controlnet_sample(
                    idx, self.single_transformer_blocks, controlnet_single_block_samples
                )
                if sample is not None:
                    hidden_states[:, encoder_hidden_states.shape[1] :, ...] = (
                        hidden_states[:, encoder_hidden_states.shape[1] :, ...] + sample
                    )

        # 7. Project output
        hidden_states = hidden_states[:, encoder_hidden_states.shape[1] :, ...]

        # Final norm with last 2 modulations
        final_modulations = pooled_temb[:, -2:]
        hidden_states = self.norm_out(hidden_states, final_modulations)
        hidden_states = self.proj_out(hidden_states)

        return hidden_states

    def _get_single_block_modulations(self, pooled_temb: mx.array, block_idx: int) -> mx.array:
        """
        Get modulations for a single transformer block.

        Single block modulations are at indices 0-113.
        Each block gets 3 modulations (shift, scale, gate).
        """
        start_idx = 3 * block_idx
        return pooled_temb[:, start_idx : start_idx + 3]

    def _get_joint_block_modulations(self, pooled_temb: mx.array, block_idx: int) -> tuple[mx.array, mx.array]:
        """
        Get modulations for a joint transformer block.

        Joint block image modulations: indices 114-227 (19 × 6)
        Joint block text modulations: indices 228-341 (19 × 6)

        Each block gets 12 modulations total (6 for image, 6 for text).
        """
        img_offset = 3 * self.num_single_transformer_blocks  # 114
        txt_offset = img_offset + 6 * self.num_transformer_blocks  # 228

        img_start = img_offset + 6 * block_idx
        txt_start = txt_offset + 6 * block_idx

        image_modulations = pooled_temb[:, img_start : img_start + 6]
        text_modulations = pooled_temb[:, txt_start : txt_start + 6]

        return image_modulations, text_modulations

    def _compute_rotary_embeddings(self, prompt_embeds: mx.array, config: Config) -> mx.array:
        """Compute rotary position embeddings for attention."""
        txt_ids = self._prepare_text_ids(seq_len=prompt_embeds.shape[1])
        img_ids = self._prepare_latent_image_ids(height=config.height, width=config.width)
        ids = mx.concatenate((txt_ids, img_ids), axis=1)
        return self.pos_embed(ids)

    @staticmethod
    def _prepare_latent_image_ids(height: int, width: int) -> mx.array:
        """Prepare latent image position IDs."""
        latent_width = width // 16
        latent_height = height // 16
        latent_image_ids = mx.zeros((latent_height, latent_width, 3))
        latent_image_ids = latent_image_ids.at[:, :, 1].add(mx.arange(0, latent_height)[:, None])
        latent_image_ids = latent_image_ids.at[:, :, 2].add(mx.arange(0, latent_width)[None, :])
        latent_image_ids = mx.repeat(latent_image_ids[None, :], 1, axis=0)
        latent_image_ids = mx.reshape(latent_image_ids, (1, latent_width * latent_height, 3))
        return latent_image_ids

    @staticmethod
    def _prepare_text_ids(seq_len: int) -> mx.array:
        """Prepare text position IDs."""
        return mx.zeros((1, seq_len, 3))

    @staticmethod
    def _get_controlnet_sample(
        idx: int,
        blocks: list,
        controlnet_samples: list[mx.array] | None,
    ) -> mx.array | None:
        """Get controlnet residual for a block if available."""
        if controlnet_samples is None or len(controlnet_samples) == 0:
            return None

        num_blocks = len(blocks)
        num_samples = len(controlnet_samples)
        interval_control = int(math.ceil(num_blocks / num_samples))
        control_index = idx // interval_control
        return controlnet_samples[control_index]
