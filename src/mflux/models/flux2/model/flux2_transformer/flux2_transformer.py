"""Main transformer for FLUX.2.

FLUX.2 transformer architecture:
- 8 joint transformer blocks (double stream: image + text)
- 48 single transformer blocks (merged stream)
- Global modulation layers (single instance, shared across all blocks)
- 128 input channels (latent from 32-channel VAE with 2x2 patches)
- 6144 hidden dim (48 heads * 128 head_dim)
- Mistral3 encoder output (joint_attention_dim = 15360)

Key architectural difference from per-block modulation:
- FLUX.2 computes modulation ONCE from conditioning and reuses for ALL blocks
- This is more efficient as the same shift/scale/gate is applied to every block
"""

import mlx.core as mx
from mlx import nn

from mflux.models.common.config.config import Config
from mflux.models.common.config.model_config import ModelConfig
from mflux.models.flux2.model.flux2_transformer.flux2_embed_nd import Flux2EmbedND
from mflux.models.flux2.model.flux2_transformer.flux2_joint_transformer_block import Flux2JointTransformerBlock
from mflux.models.flux2.model.flux2_transformer.flux2_modulation import DoubleStreamModulation, SingleStreamModulation
from mflux.models.flux2.model.flux2_transformer.flux2_single_transformer_block import Flux2SingleTransformerBlock
from mflux.models.flux2.model.flux2_transformer.flux2_time_guidance_embed import Flux2TimeGuidanceEmbed


class Flux2NormOut(nn.Module):
    """Output normalization with modulation for FLUX.2.

    Combines LayerNorm (affine=False) with learned modulation projection.
    Produces shift and scale from conditioning.
    """

    def __init__(self, hidden_dim: int = 6144):
        super().__init__()
        self.norm = nn.LayerNorm(dims=hidden_dim, eps=1e-6, affine=False)
        # norm_out.linear.weight: [12288, 6144] = [2 * hidden_dim, hidden_dim]
        self.linear = nn.Linear(hidden_dim, 2 * hidden_dim, bias=False)

    def __call__(self, hidden_states: mx.array, conditioning: mx.array) -> mx.array:
        """Apply output normalization with modulation.

        Args:
            hidden_states: Input tensor [batch, seq_len, hidden_dim]
            conditioning: Conditioning tensor [batch, hidden_dim]

        Returns:
            Normalized and modulated output [batch, seq_len, hidden_dim]
        """
        # Compute shift and scale from conditioning
        modulation = self.linear(nn.silu(conditioning))
        shift, scale = mx.split(modulation, 2, axis=-1)

        # Apply norm and modulation
        hidden_states = self.norm(hidden_states)
        hidden_states = hidden_states * (1 + scale[:, None]) + shift[:, None]

        return hidden_states


class Flux2Transformer(nn.Module):
    """FLUX.2 Transformer with 8 joint + 48 single blocks.

    Args:
        model_config: Model configuration
        num_transformer_blocks: Number of joint blocks (default 8)
        num_single_transformer_blocks: Number of single blocks (default 48)
    """

    def __init__(
        self,
        model_config: ModelConfig,
        num_transformer_blocks: int = 8,
        num_single_transformer_blocks: int = 48,
    ):
        super().__init__()
        self.hidden_dim = 6144
        self.context_dim = 15360  # joint_attention_dim from Mistral3

        # Position embeddings
        self.pos_embed = Flux2EmbedND()

        # Input projections
        # FLUX.2 has in_channels=128 (from 32-channel VAE with patches)
        self.x_embedder = nn.Linear(model_config.x_embedder_input_dim(), self.hidden_dim)
        self.context_embedder = nn.Linear(self.context_dim, self.hidden_dim)

        # Time/guidance embedding - outputs hidden_dim (6144)
        self.time_guidance_embed = Flux2TimeGuidanceEmbed(in_dim=256, hidden_dim=self.hidden_dim)

        # Global modulation layers - SINGLE instance, shared across all blocks
        # These are NOT lists - same modulation is used for every block
        self.double_stream_modulation_img = DoubleStreamModulation(self.hidden_dim)
        self.double_stream_modulation_txt = DoubleStreamModulation(self.hidden_dim)
        self.single_stream_modulation = SingleStreamModulation(self.hidden_dim)

        # Transformer blocks
        self.transformer_blocks = [
            Flux2JointTransformerBlock(i) for i in range(num_transformer_blocks)
        ]
        self.single_transformer_blocks = [
            Flux2SingleTransformerBlock(i) for i in range(num_single_transformer_blocks)
        ]

        # Output projection
        # norm_out.linear.weight: [12288, 6144] = [2 * hidden_dim, hidden_dim]
        self.norm_out = Flux2NormOut(self.hidden_dim)
        self.proj_out = nn.Linear(self.hidden_dim, 128, bias=False)  # 128 = in_channels

    def __call__(
        self,
        t: int,
        config: Config,
        hidden_states: mx.array,
        prompt_embeds: mx.array,
        pooled_prompt_embeds: mx.array,
    ) -> mx.array:
        """Forward pass through FLUX.2 transformer.

        Args:
            t: Current timestep index
            config: Configuration object
            hidden_states: Latent image states [batch, seq, in_channels]
            prompt_embeds: Text encoder output [batch, seq, context_dim]
            pooled_prompt_embeds: Pooled text embeddings [batch, pooled_dim]

        Returns:
            Predicted noise/velocity [batch, seq, in_channels]
        """
        # 1. Create embeddings
        hidden_states = self.x_embedder(hidden_states)
        encoder_hidden_states = self.context_embedder(prompt_embeds)

        # 2. Compute time/guidance conditioning
        conditioning = self._compute_conditioning(t, config)

        # 3. Compute rotary embeddings
        image_rotary_embeddings = self._compute_rotary_embeddings(prompt_embeds, config)

        # 4. Compute global modulation ONCE (shared across all blocks)
        img_modulation = self.double_stream_modulation_img(conditioning)
        txt_modulation = self.double_stream_modulation_txt(conditioning)
        single_modulation = self.single_stream_modulation(conditioning)

        # 5. Run joint transformer blocks with shared modulation
        for block in self.transformer_blocks:
            encoder_hidden_states, hidden_states = block(
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                rotary_embeddings=image_rotary_embeddings,
                img_modulation=img_modulation,
                txt_modulation=txt_modulation,
            )

        # 6. Concatenate streams for single blocks
        hidden_states = mx.concatenate([encoder_hidden_states, hidden_states], axis=1)

        # 7. Run single transformer blocks with shared modulation
        for block in self.single_transformer_blocks:
            hidden_states = block(
                hidden_states=hidden_states,
                rotary_embeddings=image_rotary_embeddings,
                modulation=single_modulation,
            )

        # 8. Extract image portion and project output
        hidden_states = hidden_states[:, encoder_hidden_states.shape[1]:, ...]

        # Apply output normalization with modulation
        hidden_states = self.norm_out(hidden_states, conditioning)
        hidden_states = self.proj_out(hidden_states)

        return hidden_states

    def _compute_conditioning(self, t: int, config: Config) -> mx.array:
        """Compute time/guidance conditioning.

        Args:
            t: Current timestep index
            config: Configuration object

        Returns:
            Conditioning tensor [batch, hidden_dim]
        """
        time_step = config.scheduler.sigmas[t] * config.num_train_steps
        time_step = mx.broadcast_to(time_step, (1,)).astype(config.precision)
        guidance = mx.broadcast_to(config.guidance * config.num_train_steps, (1,)).astype(config.precision)
        conditioning = self.time_guidance_embed(time_step, guidance)
        return conditioning

    def _compute_rotary_embeddings(self, prompt_embeds: mx.array, config: Config) -> mx.array:
        """Compute rotary position embeddings.

        Args:
            prompt_embeds: Text embeddings to get sequence length
            config: Configuration for image dimensions

        Returns:
            Rotary embeddings
        """
        txt_ids = self._prepare_text_ids(seq_len=prompt_embeds.shape[1])
        img_ids = self._prepare_latent_image_ids(height=config.height, width=config.width)
        ids = mx.concatenate((txt_ids, img_ids), axis=1)
        image_rotary_emb = self.pos_embed(ids)
        return image_rotary_emb

    @staticmethod
    def _prepare_latent_image_ids(height: int, width: int) -> mx.array:
        """Prepare position IDs for image latents.

        Args:
            height: Image height in pixels
            width: Image width in pixels

        Returns:
            Position IDs [1, seq_len, 3]
        """
        # FLUX.2 uses 32-channel VAE with 2x2 patches, so downscale factor is 16
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
        """Prepare position IDs for text tokens.

        Args:
            seq_len: Text sequence length

        Returns:
            Position IDs [1, seq_len, 3]
        """
        return mx.zeros((1, seq_len, 3))
