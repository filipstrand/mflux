"""
LongCat-Image transformer model.

Architecture:
- 10 joint transformer blocks (bi-directional attention)
- 20 single transformer blocks (self-attention on concatenated features)
- Qwen2.5-VL text encoder (3584 hidden size)
- Standard 16-channel VAE
- FlowMatch scheduler
"""

import math

import mlx.core as mx
from mlx import nn

from mflux.models.common.config.config import Config
from mflux.models.flux.model.flux_transformer.ada_layer_norm_continuous import AdaLayerNormContinuous
from mflux.models.flux.model.flux_transformer.embed_nd import EmbedND
from mflux.models.longcat.model.longcat_transformer.longcat_joint_transformer_block import LongCatJointTransformerBlock
from mflux.models.longcat.model.longcat_transformer.longcat_single_transformer_block import LongCatSingleTransformerBlock
from mflux.models.longcat.model.longcat_transformer.longcat_time_text_embed import LongCatTimeTextEmbed


class LongCatTransformer(nn.Module):
    """
    LongCat-Image Flow Match Transformer.

    Architecture details:
    - 10 joint transformer blocks for bi-directional image-text attention
    - 20 single transformer blocks for self-attention on concatenated features
    - Hidden dimension: 3072 (24 heads * 128 head dim)
    - Context dimension: 3584 (Qwen2.5-VL hidden size)
    - x_embedder input: 64 (16 latent channels * 4 for patch packing)

    Based on the FLUX Flow Match architecture but with different layer counts
    and text encoder integration.
    """

    NUM_JOINT_BLOCKS = 10
    NUM_SINGLE_BLOCKS = 20
    HIDDEN_DIM = 3072
    CONTEXT_DIM = 3584  # Qwen2.5-VL hidden size
    X_EMBEDDER_DIM = 64  # 16 latent channels * 4

    def __init__(self):
        super().__init__()
        self.pos_embed = EmbedND()
        self.x_embedder = nn.Linear(self.X_EMBEDDER_DIM, self.HIDDEN_DIM)
        self.time_text_embed = LongCatTimeTextEmbed()
        self.context_embedder = nn.Linear(self.CONTEXT_DIM, self.HIDDEN_DIM)
        self.transformer_blocks = [
            LongCatJointTransformerBlock(i) for i in range(self.NUM_JOINT_BLOCKS)
        ]
        self.single_transformer_blocks = [
            LongCatSingleTransformerBlock(i) for i in range(self.NUM_SINGLE_BLOCKS)
        ]
        self.norm_out = AdaLayerNormContinuous(self.HIDDEN_DIM, self.HIDDEN_DIM)
        self.proj_out = nn.Linear(self.HIDDEN_DIM, self.X_EMBEDDER_DIM)

    def __call__(
        self,
        t: int,
        config: Config,
        hidden_states: mx.array,
        prompt_embeds: mx.array,
        pooled_prompt_embeds: mx.array,
    ) -> mx.array:
        """
        Forward pass through LongCat transformer.

        Args:
            t: Current timestep index
            config: Model configuration
            hidden_states: Latent image features [B, H*W, C]
            prompt_embeds: Text embeddings from Qwen2.5-VL [B, seq_len, 3584]
            pooled_prompt_embeds: Pooled text embeddings [B, 3584]

        Returns:
            Predicted noise/velocity [B, H*W, 64]
        """
        # 1. Create embeddings
        hidden_states = self.x_embedder(hidden_states)
        encoder_hidden_states = self.context_embedder(prompt_embeds)
        text_embeddings = self._compute_text_embeddings(t, pooled_prompt_embeds, config)
        image_rotary_embeddings = self._compute_rotary_embeddings(prompt_embeds, config)

        # 2. Run the joint transformer blocks
        for block in self.transformer_blocks:
            encoder_hidden_states, hidden_states = block(
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                text_embeddings=text_embeddings,
                rotary_embeddings=image_rotary_embeddings,
            )

        # 3. Concat the hidden states
        hidden_states = mx.concatenate([encoder_hidden_states, hidden_states], axis=1)

        # 4. Run the single transformer blocks
        for block in self.single_transformer_blocks:
            hidden_states = block(
                hidden_states=hidden_states,
                text_embeddings=text_embeddings,
                rotary_embeddings=image_rotary_embeddings,
            )

        # 5. Project the final output
        hidden_states = hidden_states[:, encoder_hidden_states.shape[1]:, ...]
        hidden_states = self.norm_out(hidden_states, text_embeddings)
        hidden_states = self.proj_out(hidden_states)
        return hidden_states

    def _compute_text_embeddings(
        self,
        t: int,
        pooled_prompt_embeds: mx.array,
        config: Config,
    ) -> mx.array:
        """Compute combined time and text embeddings."""
        time_step = config.scheduler.sigmas[t] * config.num_train_steps
        time_step = mx.broadcast_to(time_step, (1,)).astype(config.precision)
        # LongCat doesn't use guidance in time_text_embed
        text_embeddings = self.time_text_embed(time_step, pooled_prompt_embeds)
        return text_embeddings

    def _compute_rotary_embeddings(
        self,
        prompt_embeds: mx.array,
        config: Config,
    ) -> mx.array:
        """Compute rotary position embeddings for attention."""
        txt_ids = self._prepare_text_ids(seq_len=prompt_embeds.shape[1])
        img_ids = self._prepare_latent_image_ids(height=config.height, width=config.width)
        ids = mx.concatenate((txt_ids, img_ids), axis=1)
        image_rotary_emb = self.pos_embed(ids)
        return image_rotary_emb

    @staticmethod
    def _prepare_latent_image_ids(height: int, width: int) -> mx.array:
        """Prepare position IDs for latent image patches."""
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
        """Prepare position IDs for text tokens."""
        return mx.zeros((1, seq_len, 3))
