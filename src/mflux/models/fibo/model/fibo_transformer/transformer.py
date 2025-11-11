import mlx.core as mx
from mlx import nn

from mflux.config.runtime_config import RuntimeConfig
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
        attention_head_dim: int = 128,
        num_attention_heads: int = 24,
        joint_attention_dim: int = 4096,
        text_encoder_dim: int = 2048,
        rope_theta: int = 10000,
        time_theta: int = 10000,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = in_channels
        self.num_layers = num_layers
        self.num_single_layers = num_single_layers
        self.attention_head_dim = attention_head_dim
        self.num_attention_heads = num_attention_heads
        self.joint_attention_dim = joint_attention_dim
        self.text_encoder_dim = text_encoder_dim

        self.inner_dim = num_attention_heads * attention_head_dim

        # Positional + time embeddings (BriaFibo-style RoPE)
        self.pos_embed = FiboEmbedND(theta=rope_theta)
        self.time_embed = BriaFiboTimestepProjEmbeddings(embedding_dim=self.inner_dim, time_theta=time_theta)

        # Context and latent projections
        self.context_embedder = nn.Linear(self.joint_attention_dim, self.inner_dim)
        self.x_embedder = nn.Linear(self.in_channels, self.inner_dim)

        # Transformer blocks
        self.transformer_blocks = [
            FiboJointTransformerBlock(
                dim=self.inner_dim,
                num_attention_heads=self.num_attention_heads,
                attention_head_dim=self.attention_head_dim,
            )
            for _ in range(self.num_layers)
        ]

        self.single_transformer_blocks = [
            FiboSingleTransformerBlock(
                dim=self.inner_dim,
                num_attention_heads=self.num_attention_heads,
                attention_head_dim=self.attention_head_dim,
            )
            for _ in range(self.num_single_layers)
        ]

        # Output norm + projection
        self.norm_out = AdaLayerNormContinuous(self.inner_dim, self.inner_dim)
        self.proj_out = nn.Linear(self.inner_dim, self.out_channels)

        # Text projection per layer
        caption_projection = [
            BriaFiboTextProjection(in_features=text_encoder_dim, hidden_size=self.inner_dim // 2)
            for _ in range(self.num_layers + self.num_single_layers)
        ]
        self.caption_projection = caption_projection

    def __call__(
        self,
        t: int,
        config: RuntimeConfig,
        hidden_states: mx.array,
        encoder_hidden_states: mx.array,
        text_encoder_layers: list[mx.array],
    ) -> mx.array:
        batch_size, seq_len, channels = hidden_states.shape

        # Handle classifier-free guidance: duplicate latents if encoder_hidden_states has double batch size
        encoder_batch_size = encoder_hidden_states.shape[0]
        if encoder_batch_size == batch_size * 2:
            hidden_states = mx.concatenate([hidden_states, hidden_states], axis=0)
            batch_size = hidden_states.shape[0]

        # 1. Project latents and context
        # (B, seq, C_in) -> (B, seq, inner_dim)
        hidden_states = self.x_embedder(hidden_states)
        encoder_hidden_states = self.context_embedder(encoder_hidden_states)

        # 2. Time embeddings
        timestep_value = config.scheduler.timesteps[t]
        timestep = mx.full(
            (batch_size,),
            timestep_value,
            dtype=hidden_states.dtype,
        )
        temb = self.time_embed(timestep, dtype=hidden_states.dtype)

        # 3. Prepare image and text IDs
        max_tokens = encoder_hidden_states.shape[1]
        txt_ids = self._prepare_text_ids(max_tokens=max_tokens, dtype=hidden_states.dtype)
        img_ids = self._prepare_latent_image_ids(
            height=config.height,
            width=config.width,
            dtype=hidden_states.dtype,
        )

        # 4. RoPE ids and embeddings
        if txt_ids.ndim == 3 and txt_ids.shape[0] == 1:
            txt_ids = txt_ids[0]
        if img_ids.ndim == 3 and img_ids.shape[0] == 1:
            img_ids = img_ids[0]

        ids = mx.concatenate((txt_ids, img_ids), axis=0)
        ids = mx.expand_dims(ids, axis=0)  # (1, txt+img, 3)
        image_rotary_emb = self.pos_embed(ids)

        # 5. Compute attention mask
        attention_mask = self._compute_attention_mask(
            batch_size=batch_size,
            config=config,
            encoder_hidden_states=encoder_hidden_states,
            max_tokens=max_tokens,
        )

        # 6. Project text encoder layers (convert list to projected layers)
        projected_text_layers: list[mx.array] = []
        for i, text_layer in enumerate(text_encoder_layers):
            projected = self.caption_projection[i](text_layer)
            projected_text_layers.append(projected)
        text_encoder_layers = projected_text_layers

        # 7. Joint transformer blocks
        block_id = 0
        for index_block, block in enumerate(self.transformer_blocks):
            current_text_layer = text_encoder_layers[block_id]
            # Split encoder_hidden_states into context and text halves for clarity
            ctx_half = encoder_hidden_states[:, :, : self.inner_dim // 2]
            txt_half = current_text_layer
            encoder_hidden_states = mx.concatenate(
                [
                    ctx_half,
                    txt_half,
                ],
                axis=-1,
            )
            block_id += 1

            encoder_hidden_states, hidden_states = block(
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                temb=temb,
                image_rotary_emb=image_rotary_emb,
                attention_mask=attention_mask,
            )

        # 8. Single transformer blocks
        for block in self.single_transformer_blocks:
            current_text_layer = text_encoder_layers[block_id]
            encoder_hidden_states = mx.concatenate(
                [
                    encoder_hidden_states[:, :, : self.inner_dim // 2],
                    current_text_layer,
                ],
                axis=-1,
            )
            block_id += 1

            # Concatenate encoder + hidden along sequence dim
            combined = mx.concatenate([encoder_hidden_states, hidden_states], axis=1)
            combined = block(
                hidden_states=combined,
                temb=temb,
                image_rotary_emb=image_rotary_emb,
                attention_mask=attention_mask,
            )

            # Split back into encoder and hidden streams
            encoder_len = encoder_hidden_states.shape[1]
            encoder_hidden_states = combined[:, :encoder_len, ...]
            hidden_states = combined[:, encoder_len:, ...]

        # 9. Output projection back to latent channels
        hidden_states = self.norm_out(hidden_states, temb)
        output = self.proj_out(hidden_states)  # (B, seq, out_channels)
        return output

    @staticmethod
    def _prepare_text_ids(max_tokens: int, dtype=mx.float32) -> mx.array:
        return mx.zeros((max_tokens, 3), dtype=dtype)

    @staticmethod
    def _prepare_latent_image_ids(height: int, width: int, dtype=mx.float32) -> mx.array:
        vae_scale_factor = 16
        latent_height = height // vae_scale_factor
        latent_width = width // vae_scale_factor
        row_indices = mx.arange(0, latent_height, dtype=dtype)[:, None]  # (latent_height, 1)
        row_indices = mx.broadcast_to(row_indices, (latent_height, latent_width))  # (latent_height, latent_width)
        col_indices = mx.arange(0, latent_width, dtype=dtype)[None, :]  # (1, latent_width)
        col_indices = mx.broadcast_to(col_indices, (latent_height, latent_width))  # (latent_height, latent_width)
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

    def _compute_attention_mask(
        self,
        batch_size: int,
        config: RuntimeConfig,
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
        attention_mask = self._prepare_attention_mask(attention_mask_2d)
        attention_mask = attention_mask.astype(encoder_hidden_states.dtype)
        return attention_mask
