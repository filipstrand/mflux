import mlx.core as mx
from mlx import nn

from mflux.models.common.config.config import Config
from mflux.models.common.config.model_config import ModelConfig
from mflux.models.flux.model.flux_transformer.ada_layer_norm_continuous import AdaLayerNormContinuous
from mflux.models.flux.model.flux_transformer.embed_nd import EmbedND
from mflux.models.flux.model.flux_transformer.single_transformer_block import SingleTransformerBlock
from mflux.models.flux.model.flux_transformer.time_text_embed import TimeTextEmbed
from mflux.models.flux.model.flux_transformer.transformer import Transformer
from mflux.models.flux.variants.concept_attention.attention_data import TimestepAttentionData
from mflux.models.flux.variants.concept_attention.joint_transformer_block_concept import JointTransformerBlockConcept


class TransformerConcept(nn.Module):
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
        self.transformer_blocks = [JointTransformerBlockConcept(i) for i in range(num_transformer_blocks)]
        self.single_transformer_blocks = [SingleTransformerBlock(i) for i in range(num_single_transformer_blocks)]
        self.norm_out = AdaLayerNormContinuous(3072, 3072)
        self.proj_out = nn.Linear(3072, 64)

    def __call__(
        self,
        t: int,
        config: Config,
        hidden_states: mx.array,
        prompt_embeds: mx.array,
        prompt_embeds_concept: mx.array,
        pooled_prompt_embeds: mx.array,
        pooled_prompt_embeds_concept: mx.array,
    ) -> tuple[mx.array, TimestepAttentionData]:
        # 1. Create embeddings
        hidden_states = self.x_embedder(hidden_states)
        encoder_hidden_states = self.context_embedder(prompt_embeds)
        encoder_hidden_states_concept = self.context_embedder(prompt_embeds_concept)
        text_embeddings = Transformer.compute_text_embeddings(t, pooled_prompt_embeds, self.time_text_embed, config)  # fmt: off
        text_embeddings_concept = Transformer.compute_text_embeddings(t, pooled_prompt_embeds_concept, self.time_text_embed, config)  # fmt: off
        image_rotary_embeddings = Transformer.compute_rotary_embeddings(prompt_embeds, self.pos_embed, config, None)  # fmt: off
        image_rotary_embeddings_concept = Transformer.compute_rotary_embeddings(prompt_embeds_concept, self.pos_embed, config, None)  # fmt: off

        # 2. Run the joint flux_transformer blocks
        attention_information = []
        for idx, block in enumerate(self.transformer_blocks):
            encoder_hidden_states, hidden_states, encoder_hidden_states_concept, attn = block(
                layer_idx=idx,
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                encoder_hidden_states_concept=encoder_hidden_states_concept,
                text_embeddings=text_embeddings,
                text_embeddings_concept=text_embeddings_concept,
                rotary_embeddings=image_rotary_embeddings,
                rotary_embeddings_concept=image_rotary_embeddings_concept,
            )
            attention_information.append(attn)

        # 3. Concat the hidden states
        hidden_states = mx.concatenate([encoder_hidden_states, hidden_states], axis=1)

        # 4. Run the single flux_transformer blocks
        for idx, block in enumerate(self.single_transformer_blocks):
            hidden_states = block(
                hidden_states=hidden_states,
                text_embeddings=text_embeddings,
                rotary_embeddings=image_rotary_embeddings,
            )

        # 5. Project the final output
        hidden_states = hidden_states[:, encoder_hidden_states.shape[1] :, ...]
        hidden_states = self.norm_out(hidden_states, text_embeddings)
        hidden_states = self.proj_out(hidden_states)

        # 6. Create timestep attention data structure
        timestep_attention = TimestepAttentionData(
            t=t,
            attention_information=attention_information,
        )

        return hidden_states, timestep_attention
