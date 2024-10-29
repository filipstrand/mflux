import mlx.core as mx
from mlx import nn

from mflux.config.model_config import ModelConfig
from mflux.config.runtime_config import RuntimeConfig
from mflux.models.transformer.embed_nd import EmbedND
from mflux.models.transformer.joint_transformer_block import (
    JointTransformerBlock,
)
from mflux.models.transformer.single_transformer_block import (
    SingleTransformerBlock,
)
from mflux.models.transformer.time_text_embed import TimeTextEmbed
from mflux.models.transformer.transformer import Transformer


class TransformerControlnet(nn.Module):
    def __init__(
        self,
        model_config: ModelConfig,
        num_blocks: int,
        num_single_blocks: int,
    ):
        super().__init__()
        self.pos_embed = EmbedND()
        self.x_embedder = nn.Linear(64, 3072)
        self.time_text_embed = TimeTextEmbed(model_config=model_config)
        self.context_embedder = nn.Linear(4096, 3072)
        self.transformer_blocks = [JointTransformerBlock(i) for i in range(num_blocks)]
        self.single_transformer_blocks = [SingleTransformerBlock(i) for i in range(num_single_blocks)]

        zero_init = nn.init.constant(0)
        self.controlnet_x_embedder = nn.Linear(64, 3072).apply(zero_init)
        self.controlnet_blocks = [nn.Linear(3072, 3072).apply(zero_init) for _ in range(num_blocks)]

        self.controlnet_single_blocks = [nn.Linear(3072, 3072) for _ in range(num_single_blocks)]

    def forward(
        self,
        t: int,
        prompt_embeds: mx.array,
        pooled_prompt_embeds: mx.array,
        hidden_states: mx.array,
        controlnet_cond: mx.array,
        config: RuntimeConfig,
    ) -> (list[mx.array], list[mx.array]):
        time_step = config.sigmas[t] * config.num_train_steps
        time_step = mx.broadcast_to(time_step, (1,)).astype(config.precision)
        hidden_states = self.x_embedder(hidden_states)
        hidden_states = hidden_states + self.controlnet_x_embedder(controlnet_cond)
        conditioning_scale = config.config.controlnet_strength

        guidance = mx.broadcast_to(config.guidance * config.num_train_steps, (1,)).astype(config.precision)
        text_embeddings = self.time_text_embed.forward(time_step, pooled_prompt_embeds, guidance)
        encoder_hidden_states = self.context_embedder(prompt_embeds)
        txt_ids = Transformer.prepare_text_ids(seq_len=prompt_embeds.shape[1])
        img_ids = Transformer.prepare_latent_image_ids(height=config.height, width=config.width)
        ids = mx.concatenate((txt_ids, img_ids), axis=1)
        image_rotary_emb = self.pos_embed.forward(ids)

        block_samples = ()
        for block in self.transformer_blocks:
            encoder_hidden_states, hidden_states = block.forward(
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                text_embeddings=text_embeddings,
                rotary_embeddings=image_rotary_emb,
            )
            block_samples = block_samples + (hidden_states,)

        hidden_states = mx.concatenate([encoder_hidden_states, hidden_states], axis=1)

        # controlnet block
        controlnet_block_samples = ()
        for block_sample, controlnet_block in zip(block_samples, self.controlnet_blocks):
            block_sample = controlnet_block(block_sample)
            controlnet_block_samples = controlnet_block_samples + (block_sample,)

        single_block_samples = ()
        for block in self.single_transformer_blocks:
            hidden_states = block.forward(
                hidden_states=hidden_states,
                text_embeddings=text_embeddings,
                rotary_embeddings=image_rotary_emb,
            )
            single_block_samples = single_block_samples + (hidden_states[:, encoder_hidden_states.shape[1] :],)

        controlnet_single_block_samples = ()
        for single_block_sample, controlnet_block in zip(single_block_samples, self.controlnet_single_blocks):
            single_block_sample = controlnet_block(single_block_sample)
            controlnet_single_block_samples = controlnet_single_block_samples + (single_block_sample,)

        # scaling
        controlnet_block_samples = [sample * conditioning_scale for sample in controlnet_block_samples]
        controlnet_single_block_samples = [sample * conditioning_scale for sample in controlnet_single_block_samples]

        return controlnet_block_samples, controlnet_single_block_samples
