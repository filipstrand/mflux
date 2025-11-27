import math

import mlx.core as mx
from mlx import nn

from mflux.models.common.config.config import Config
from mflux.models.common.config.model_config import ModelConfig
from mflux.models.flux.model.flux_transformer.ada_layer_norm_continuous import AdaLayerNormContinuous
from mflux.models.flux.model.flux_transformer.embed_nd import EmbedND
from mflux.models.flux.model.flux_transformer.joint_transformer_block import JointTransformerBlock
from mflux.models.flux.model.flux_transformer.single_transformer_block import SingleTransformerBlock
from mflux.models.flux.model.flux_transformer.time_text_embed import TimeTextEmbed


class Transformer(nn.Module):
    def __init__(
        self,
        model_config: ModelConfig,
        num_transformer_blocks: int = 19,
        num_single_transformer_blocks: int = 38,
    ):
        super().__init__()
        self.pos_embed = EmbedND()
        self.x_embedder = nn.Linear(model_config.x_embedder_input_dim(), 3072)
        self.time_text_embed = TimeTextEmbed(model_config=model_config)
        self.context_embedder = nn.Linear(4096, 3072)
        self.transformer_blocks = [JointTransformerBlock(i) for i in range(num_transformer_blocks)]
        self.single_transformer_blocks = [SingleTransformerBlock(i) for i in range(num_single_transformer_blocks)]
        self.norm_out = AdaLayerNormContinuous(3072, 3072)
        self.proj_out = nn.Linear(3072, 64)

    def __call__(
        self,
        t: int,
        config: Config,
        hidden_states: mx.array,
        prompt_embeds: mx.array,
        pooled_prompt_embeds: mx.array,
        controlnet_block_samples: list[mx.array] | None = None,
        controlnet_single_block_samples: list[mx.array] | None = None,
        kontext_image_ids: mx.array | None = None,
    ) -> mx.array:
        # 1. Create embeddings
        hidden_states = self.x_embedder(hidden_states)
        encoder_hidden_states = self.context_embedder(prompt_embeds)
        text_embeddings = Transformer.compute_text_embeddings(t, pooled_prompt_embeds, self.time_text_embed, config)
        image_rotary_embeddings = Transformer.compute_rotary_embeddings(prompt_embeds, self.pos_embed, config, kontext_image_ids)  # fmt: off

        # 2. Run the joint flux_transformer blocks
        for idx, block in enumerate(self.transformer_blocks):
            encoder_hidden_states, hidden_states = self._apply_joint_transformer_block(
                idx=idx,
                block=block,
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                text_embeddings=text_embeddings,
                image_rotary_embeddings=image_rotary_embeddings,
                controlnet_block_samples=controlnet_block_samples,
            )

        # 3. Concat the hidden states
        hidden_states = mx.concatenate([encoder_hidden_states, hidden_states], axis=1)

        # 4. Run the single flux_transformer blocks
        for idx, block in enumerate(self.single_transformer_blocks):
            hidden_states = self._apply_single_transformer_block(
                idx=idx,
                block=block,
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                text_embeddings=text_embeddings,
                image_rotary_embeddings=image_rotary_embeddings,
                controlnet_single_block_samples=controlnet_single_block_samples,
            )

        # 5. Project the final output
        hidden_states = hidden_states[:, encoder_hidden_states.shape[1] :, ...]
        hidden_states = self.norm_out(hidden_states, text_embeddings)
        hidden_states = self.proj_out(hidden_states)
        return hidden_states

    def _apply_single_transformer_block(
        self,
        idx: int,
        block: SingleTransformerBlock,
        hidden_states: mx.array,
        encoder_hidden_states: mx.array,
        text_embeddings: mx.array,
        image_rotary_embeddings: mx.array,
        controlnet_single_block_samples: list[mx.array],
    ) -> mx.array:
        # 1. Apply single flux_transformer block
        hidden_states = block(
            hidden_states=hidden_states,
            text_embeddings=text_embeddings,
            rotary_embeddings=image_rotary_embeddings,
        )

        # 2. Apply previously calculated controlnet result (if applicable)
        sample = Transformer._get_controlnet_sample(idx, self.single_transformer_blocks, controlnet_single_block_samples)  # fmt: off
        hidden_states[:, encoder_hidden_states.shape[1] :, ...] += sample if sample is not None else 0

        return hidden_states

    def _apply_joint_transformer_block(
        self,
        idx: int,
        block: JointTransformerBlock,
        hidden_states: mx.array,
        encoder_hidden_states: mx.array,
        text_embeddings: mx.array,
        image_rotary_embeddings: mx.array,
        controlnet_block_samples: list[mx.array],
    ) -> mx.array:
        # 1. Apply joint flux_transformer block
        encoder_hidden_states, hidden_states = block(
            hidden_states=hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            text_embeddings=text_embeddings,
            rotary_embeddings=image_rotary_embeddings,
        )

        # 2. Apply previously calculated controlnet result (if applicable)
        sample = Transformer._get_controlnet_sample(idx, self.transformer_blocks, controlnet_block_samples)
        hidden_states += sample if sample is not None else 0

        return encoder_hidden_states, hidden_states

    @staticmethod
    def compute_rotary_embeddings(
        prompt_embeds: mx.array,
        pos_embed: EmbedND,
        config: Config,
        kontext_image_ids: mx.array | None = None,
    ) -> mx.array:
        txt_ids = Transformer._prepare_text_ids(seq_len=prompt_embeds.shape[1])
        img_ids = Transformer._prepare_latent_image_ids(height=config.height, width=config.width)

        if kontext_image_ids is not None:
            img_ids = mx.concatenate([img_ids, kontext_image_ids], axis=1)

        ids = mx.concatenate((txt_ids, img_ids), axis=1)
        image_rotary_emb = pos_embed(ids)
        return image_rotary_emb

    @staticmethod
    def compute_text_embeddings(
        t: int,
        pooled_prompt_embeds: mx.array,
        time_text_embed: TimeTextEmbed,
        config: Config,
    ) -> mx.array:
        time_step = config.scheduler.sigmas[t] * config.num_train_steps
        time_step = mx.broadcast_to(time_step, (1,)).astype(config.precision)
        guidance = mx.broadcast_to(config.guidance * config.num_train_steps, (1,)).astype(config.precision)
        text_embeddings = time_text_embed(time_step, pooled_prompt_embeds, guidance)
        return text_embeddings

    @staticmethod
    def _prepare_latent_image_ids(height: int, width: int) -> mx.array:
        latent_width = width // 16
        latent_height = height // 16
        latent_image_ids = mx.zeros((latent_height, latent_width, 3))
        latent_image_ids = latent_image_ids.at[:, :, 1].add(mx.arange(0, latent_height)[:, None])
        latent_image_ids = latent_image_ids.at[:, :, 2].add(mx.arange(0, latent_width)[None, :])
        latent_image_ids = mx.repeat(latent_image_ids[None, :], 1, axis=0)
        latent_image_ids = mx.reshape(latent_image_ids, (1, latent_width * latent_height, 3))
        return latent_image_ids

    @staticmethod
    def _prepare_text_ids(seq_len: mx.array) -> mx.array:
        return mx.zeros((1, seq_len, 3))

    @staticmethod
    def _get_controlnet_sample(
        idx: int,
        blocks: mx.array,
        controlnet_samples: list[mx.array] | None,
    ) -> mx.array | None:
        if controlnet_samples is None:
            return None

        if len(controlnet_samples) == 0:
            return None

        num_blocks = len(blocks)
        num_samples = len(controlnet_samples)
        interval_control = int(math.ceil(num_blocks / num_samples))
        control_index = idx // interval_control
        return controlnet_samples[control_index]
