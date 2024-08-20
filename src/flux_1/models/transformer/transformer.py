import mlx.core as mx
from mlx import nn

from flux_1.config.config import Config
from flux_1.config.runtime_config import RuntimeConfig
from flux_1.models.transformer.ada_layer_norm_continous import AdaLayerNormContinuous
from flux_1.models.transformer.embed_nd import EmbedND
from flux_1.models.transformer.joint_transformer_block import JointTransformerBlock
from flux_1.models.transformer.single_transformer_block import SingleTransformerBlock
from flux_1.models.transformer.time_text_embed import TimeTextEmbed


class Transformer(nn.Module):

    def __init__(self, weights: dict):
        super().__init__()
        self.pos_embed = EmbedND()
        self.x_embedder = nn.Linear(64, 3072)
        with_guidance_embed = "guidance_embedder" in weights["time_text_embed"].keys()
        self.time_text_embed = TimeTextEmbed(with_guidance_embed=with_guidance_embed)
        self.context_embedder = nn.Linear(4096, 3072)
        self.transformer_blocks = [JointTransformerBlock(i) for i in range(19)]
        self.single_transformer_blocks = [SingleTransformerBlock(i) for i in range(38)]
        self.norm_out = AdaLayerNormContinuous(3072, 3072)
        self.proj_out = nn.Linear(3072, 64)

        # Load the weights after all components are initialized
        self.update(weights)

    def predict(
            self,
            t: int,
            prompt_embeds: mx.array,
            pooled_prompt_embeds: mx.array,
            hidden_states: mx.array,
            config: RuntimeConfig,
    ) -> mx.array:
        time_step = config.sigmas[t] * config.num_train_steps
        time_step = mx.broadcast_to(time_step, (1,)).astype(config.precision)
        hidden_states = self.x_embedder(hidden_states)
        guidance = mx.broadcast_to(config.guidance * config.num_train_steps, (1,)).astype(config.precision)
        text_embeddings = self.time_text_embed.forward(time_step, pooled_prompt_embeds, guidance)
        encoder_hidden_states = self.context_embedder(prompt_embeds)
        txt_ids = Transformer._prepare_text_ids(seq_len=prompt_embeds.shape[1])
        img_ids = Transformer._prepare_latent_image_ids(config.height, config.width)
        ids = mx.concatenate((txt_ids, img_ids), axis=1)
        image_rotary_emb = self.pos_embed.forward(ids)

        for block in self.transformer_blocks:
            encoder_hidden_states, hidden_states = block.forward(
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                text_embeddings=text_embeddings,
                rotary_embeddings=image_rotary_emb
            )

        hidden_states = mx.concatenate([encoder_hidden_states, hidden_states], axis=1)

        for block in self.single_transformer_blocks:
            hidden_states = block.forward(
                hidden_states=hidden_states,
                text_embeddings=text_embeddings,
                rotary_embeddings=image_rotary_emb
            )

        hidden_states = hidden_states[:, encoder_hidden_states.shape[1]:, ...]
        hidden_states = self.norm_out.forward(hidden_states, text_embeddings)
        hidden_states = self.proj_out(hidden_states)
        noise = hidden_states
        return noise

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
