import mlx.core as mx
from mlx import nn

from mflux.models.common.config.config import Config
from mflux.models.fibo.model.fibo_transformer.fibo_embed_nd import FiboEmbedND
from mflux.models.fibo.model.fibo_transformer.joint_transformer_block import FiboJointTransformerBlock
from mflux.models.fibo.model.fibo_transformer.single_transformer_block import FiboSingleTransformerBlock
from mflux.models.fibo.model.fibo_transformer.text_projection import BriaFiboTextProjection
from mflux.models.fibo.model.fibo_transformer.time_embed import BriaFiboTimestepProjEmbeddings
from mflux.models.flux.model.flux_transformer.ada_layer_norm_continuous import AdaLayerNormContinuous


class FiboTransformer(nn.Module):
    def __init__(
        self,
        in_channels: int = 48,
        num_layers: int = 8,
        num_single_layers: int = 38,
    ):
        super().__init__()
        self.pos_embed = FiboEmbedND()
        self.x_embedder = nn.Linear(in_channels, 3072)
        self.time_embed = BriaFiboTimestepProjEmbeddings()
        self.context_embedder = nn.Linear(4096, 3072)
        self.transformer_blocks = [FiboJointTransformerBlock(i) for i in range(num_layers)]
        self.single_transformer_blocks = [FiboSingleTransformerBlock(i) for i in range(num_single_layers)]
        self.norm_out = AdaLayerNormContinuous(3072, 3072)
        self.proj_out = nn.Linear(3072, in_channels)
        self.caption_projection = [BriaFiboTextProjection() for _ in range(num_layers + num_single_layers)]

    def __call__(
        self,
        t: int,
        config: Config,
        hidden_states: mx.array,
        encoder_hidden_states: mx.array,
        text_encoder_layers: list[mx.array],
    ) -> mx.array:
        # 1. Create embeddings
        hidden_states = FiboTransformer._handle_classifier_free_guidance(hidden_states, encoder_hidden_states)
        hidden_states = self.x_embedder(hidden_states)
        encoder_hidden_states = self.context_embedder(encoder_hidden_states)
        time_embeddings = FiboTransformer._compute_time_embeddings(t, config, hidden_states.shape[0], hidden_states.dtype, self.time_embed)  # fmt: off
        image_rotary_emb = FiboTransformer._compute_rotary_embeddings(encoder_hidden_states, self.pos_embed, config, hidden_states.dtype)  # fmt: off

        # 2. Compute attention mask
        attention_mask = FiboTransformer._compute_attention_mask(
            config=config,
            batch_size=hidden_states.shape[0],
            encoder_hidden_states=encoder_hidden_states,
            max_tokens=encoder_hidden_states.shape[1],
        )

        # 3. Project the fibo-specific text encoder layers
        text_encoder_layers = [
            self.caption_projection[i](text_layer)
            for i, text_layer in enumerate(text_encoder_layers)
        ]  # fmt: off

        # 4. Run the joint transformer blocks
        block_id = 0
        for _, block in enumerate(self.transformer_blocks):
            encoder_hidden_states, hidden_states = FiboTransformer._apply_joint_transformer_block(
                block=block,
                time_embeddings=time_embeddings,
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                text_encoder_layer=text_encoder_layers[block_id],
                image_rotary_emb=image_rotary_emb,
                attention_mask=attention_mask,
            )
            block_id += 1

        # 5. Run the single transformer blocks
        for _, block in enumerate(self.single_transformer_blocks):
            encoder_hidden_states, hidden_states = FiboTransformer._apply_single_transformer_block(
                block=block,
                time_embeddings=time_embeddings,
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                text_encoder_layer=text_encoder_layers[block_id],
                image_rotary_emb=image_rotary_emb,
                attention_mask=attention_mask,
            )
            block_id += 1

        # 6. Project the final output
        hidden_states = self.norm_out(hidden_states, time_embeddings)
        hidden_states = self.proj_out(hidden_states)
        return hidden_states

    @staticmethod
    def _apply_joint_transformer_block(
        time_embeddings: mx.array,
        block: FiboJointTransformerBlock,
        hidden_states: mx.array,
        encoder_hidden_states: mx.array,
        text_encoder_layer: mx.array,
        image_rotary_emb: mx.array,
        attention_mask: mx.array,
    ) -> tuple[mx.array, mx.array]:
        encoder_hidden_states_half = encoder_hidden_states[:, :, :1536]
        encoder_hidden_states = mx.concatenate([encoder_hidden_states_half, text_encoder_layer], axis=-1)

        encoder_hidden_states, hidden_states = block(
            temb=time_embeddings,
            hidden_states=hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            image_rotary_emb=image_rotary_emb,
            attention_mask=attention_mask,
        )

        return encoder_hidden_states, hidden_states

    @staticmethod
    def _apply_single_transformer_block(
        time_embeddings: mx.array,
        block: FiboSingleTransformerBlock,
        hidden_states: mx.array,
        encoder_hidden_states: mx.array,
        text_encoder_layer: mx.array,
        image_rotary_emb: mx.array,
        attention_mask: mx.array,
    ) -> tuple[mx.array, mx.array]:
        encoder_hidden_states_half = encoder_hidden_states[:, :, :1536]
        encoder_hidden_states = mx.concatenate([encoder_hidden_states_half, text_encoder_layer], axis=-1)
        combined = mx.concatenate([encoder_hidden_states, hidden_states], axis=1)

        combined = block(
            temb=time_embeddings,
            hidden_states=combined,
            image_rotary_emb=image_rotary_emb,
            attention_mask=attention_mask,
        )

        encoder_len = encoder_hidden_states.shape[1]
        encoder_hidden_states = combined[:, :encoder_len, ...]
        hidden_states = combined[:, encoder_len:, ...]

        return encoder_hidden_states, hidden_states

    @staticmethod
    def _handle_classifier_free_guidance(hidden_states: mx.array, encoder_hidden_states: mx.array) -> mx.array:
        batch_size = hidden_states.shape[0]
        encoder_batch_size = encoder_hidden_states.shape[0]
        if encoder_batch_size == batch_size * 2:
            hidden_states = mx.concatenate([hidden_states, hidden_states], axis=0)
        return hidden_states

    @staticmethod
    def _compute_time_embeddings(
        t: int,
        config: Config,
        batch_size: int,
        dtype: mx.Dtype,
        time_embed: BriaFiboTimestepProjEmbeddings,
    ) -> mx.array:
        timestep_value = config.scheduler.timesteps[t]
        timestep = mx.full((batch_size,), timestep_value, dtype=dtype)
        return time_embed(timestep, dtype=dtype)

    @staticmethod
    def _compute_rotary_embeddings(
        encoder_hidden_states: mx.array,
        pos_embed: FiboEmbedND,
        config: Config,
        dtype: mx.Dtype,
    ) -> mx.array:
        max_tokens = encoder_hidden_states.shape[1]
        txt_ids = mx.zeros((max_tokens, 3), dtype=dtype)
        img_ids = FiboTransformer._prepare_latent_image_ids(height=config.height, width=config.width, dtype=dtype)

        if txt_ids.ndim == 3 and txt_ids.shape[0] == 1:
            txt_ids = txt_ids[0]
        if img_ids.ndim == 3 and img_ids.shape[0] == 1:
            img_ids = img_ids[0]

        ids = mx.concatenate((txt_ids, img_ids), axis=0)
        ids = mx.expand_dims(ids, axis=0)
        return pos_embed(ids)

    @staticmethod
    def _prepare_latent_image_ids(height: int, width: int, dtype=mx.float32) -> mx.array:
        vae_scale_factor = 16
        latent_height = height // vae_scale_factor
        latent_width = width // vae_scale_factor
        row_indices = mx.arange(0, latent_height, dtype=dtype)[:, None]
        row_indices = mx.broadcast_to(row_indices, (latent_height, latent_width))
        col_indices = mx.arange(0, latent_width, dtype=dtype)[None, :]
        col_indices = mx.broadcast_to(col_indices, (latent_height, latent_width))
        zeros_channel = mx.zeros((latent_height, latent_width), dtype=dtype)
        latent_image_ids = mx.stack([zeros_channel, row_indices, col_indices], axis=-1)
        latent_image_ids = mx.reshape(latent_image_ids, (latent_height * latent_width, 3))
        return latent_image_ids

    @staticmethod
    def _prepare_attention_mask(attention_mask_2d: mx.array) -> mx.array:
        attention_matrix = mx.einsum("bi,bj->bij", attention_mask_2d, attention_mask_2d)
        mask_dtype = attention_mask_2d.dtype
        min_dtype_value = mx.finfo(mask_dtype).min
        attention_matrix = mx.where(
            attention_matrix == 1,
            mx.zeros_like(attention_matrix).astype(mask_dtype),
            (mx.ones_like(attention_matrix) * min_dtype_value).astype(mask_dtype),
        )
        attention_matrix = mx.expand_dims(attention_matrix, axis=1)
        return attention_matrix

    @staticmethod
    def _compute_attention_mask(
        batch_size: int,
        config: Config,
        encoder_hidden_states: mx.array,
        max_tokens: int,
    ) -> mx.array:
        vae_scale_factor = 16
        latent_height = config.height // vae_scale_factor
        latent_width = config.width // vae_scale_factor
        latent_seq_len = latent_height * latent_width
        prompt_attention_mask = mx.ones((batch_size, max_tokens), dtype=mx.float32)
        latent_attention_mask = mx.ones((batch_size, latent_seq_len), dtype=mx.float32)
        attention_mask_2d = mx.concatenate([prompt_attention_mask, latent_attention_mask], axis=1)
        attention_mask = FiboTransformer._prepare_attention_mask(attention_mask_2d)
        attention_mask = attention_mask.astype(encoder_hidden_states.dtype)
        return attention_mask
