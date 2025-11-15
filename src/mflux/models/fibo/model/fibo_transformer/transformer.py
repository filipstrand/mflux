import mlx.core as mx
from mlx import nn

from mflux.models.fibo.model.fibo_transformer.fibo_embed_nd import FiboEmbedND
from mflux.models.fibo.model.fibo_transformer.joint_transformer_block import FiboJointTransformerBlock
from mflux.models.fibo.model.fibo_transformer.single_transformer_block import FiboSingleTransformerBlock
from mflux.models.fibo.model.fibo_transformer.text_projection import BriaFiboTextProjection
from mflux.models.fibo.model.fibo_transformer.time_embed import BriaFiboTimestepProjEmbeddings
from mflux.models.flux.model.flux_transformer.ada_layer_norm_continuous import AdaLayerNormContinuous
from mflux_debugger.semantic_checkpoint import debug_checkpoint


class FiboTransformer(nn.Module):
    """
    MLX implementation of the BriaFiboTransformer2DModel core math.

    Notes / assumptions:
    - IP-Adapter specific branches are intentionally ignored for now.
    - Configuration matches the default diffusers FIBO config:
      patch_size=1, in_channels=64, num_layers=19, num_single_layers=38,
      num_attention_heads=24, attention_head_dim=128, joint_attention_dim=4096,
      text_encoder_dim=2048, axes_dims_rope=[16, 56, 56], rope_theta=10000, time_theta=10000.
    """

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
        hidden_states: mx.array,
        encoder_hidden_states: mx.array,
        text_encoder_layers: list[mx.array],
        timestep: mx.array,
        img_ids: mx.array,
        txt_ids: mx.array,
        guidance: mx.array | None = None,
        attention_mask: mx.array | None = None,
    ) -> mx.array:
        """
        Args:
            hidden_states: (batch, channels, height, width)
            encoder_hidden_states: (batch, seq_len, joint_attention_dim)
            text_encoder_layers: list of (batch, seq_len, text_encoder_dim)
            timestep: (batch,) scalar timestep per batch element
            img_ids: (batch, img_seq, 3)
            txt_ids: (batch, txt_seq, 3)
            guidance: (batch,) optional guidance for future extension (ignored for now)
        Returns:
            Sample tensor with same spatial shape and channel count as input hidden_states.
        """
        # ----- MLX checkpoint A: transformer forward entry -----
        debug_checkpoint("mlx_A", skip=False)
        # PyTorch FIBO transformer expects hidden_states with shape (B, seq_len, in_channels)
        batch_size, seq_len, channels = hidden_states.shape

        # 1. Project latents and context
        # (B, seq, C_in) -> (B, seq, inner_dim)
        hidden_states = self.x_embedder(hidden_states)
        encoder_hidden_states = self.context_embedder(encoder_hidden_states)

        # 2. Time embeddings (guidance branch reserved for future use)
        timestep = timestep.astype(hidden_states.dtype)
        temb = self.time_embed(timestep, dtype=hidden_states.dtype)

        # 3. RoPE ids and embeddings
        if txt_ids.ndim == 3 and txt_ids.shape[0] == 1:
            txt_ids = txt_ids[0]
        if img_ids.ndim == 3 and img_ids.shape[0] == 1:
            img_ids = img_ids[0]

        ids = mx.concatenate((txt_ids, img_ids), axis=0)
        ids = mx.expand_dims(ids, axis=0)  # (1, txt+img, 3)
        image_rotary_emb = self.pos_embed(ids)

        # 4. Project text encoder layers
        projected_text_layers: list[mx.array] = []
        for i, text_layer in enumerate(text_encoder_layers):
            projected = self.caption_projection[i](text_layer)
            projected_text_layers.append(projected)
        text_encoder_layers = projected_text_layers

        # 5. Joint transformer blocks
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

        # 6. Single transformer blocks
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

        # 7. Output projection back to latent channels
        # ----- MLX checkpoint B: just before final norm/projection -----
        debug_checkpoint("mlx_B", skip=False)

        hidden_states = self.norm_out(hidden_states, temb)
        output = self.proj_out(hidden_states)  # (B, seq, out_channels)
        return output
