"""Qwen3-VL 2B Encoder for embedding and reranking models.

This encoder is configured for the 2B model variants:
- Qwen3-VL-Embedding-2B
- Qwen3-VL-Reranker-2B

Key differences from 7B model:
- hidden_size: 2048 (vs 3584)
- num_attention_heads: 16 (vs 28)
- intermediate_size: 8192 (vs 18944)
- Vision: 24 blocks (vs 32), LayerNorm (vs RMSNorm)
"""

import mlx.core as mx
from mlx import nn

from mflux.models.qwen.model.qwen_text_encoder.qwen_rms_norm import QwenRMSNorm
from mflux.models.qwen.model.qwen_text_encoder.qwen_rope import QwenRotaryEmbedding

from .qwen3_vl_2b_encoder_layer import Qwen3VL2BEncoderLayer
from .qwen3_vl_vision import Qwen3VL2BVisionTransformer


class Qwen3VL2BEncoder(nn.Module):
    """Encoder for Qwen3-VL 2B models.

    Supports both embedding and reranking use cases.
    The vision transformer is optional and loaded separately.
    """

    # 2B model configuration (from HuggingFace config.json)
    VOCAB_SIZE = 151936  # Actual vocab size from weights
    HIDDEN_SIZE = 2048
    NUM_HIDDEN_LAYERS = 28
    NUM_ATTENTION_HEADS = 16
    NUM_KEY_VALUE_HEADS = 8  # 8 (not 4) - from k_proj shape (1024, 2048)
    INTERMEDIATE_SIZE = 8192
    MAX_POSITION_EMBEDDINGS = 128000
    ROPE_THETA = 1000000.0
    RMS_NORM_EPS = 1e-6

    # Vision configuration for 2B models
    VISION_EMBED_DIM = 1024  # 2B uses 1024 (7B uses 1280)
    VISION_DEPTH = 24  # 2B uses 24 blocks (not 32)
    VISION_NUM_HEADS = 16
    VISION_MLP_RATIO = 4.0  # Standard 4x (not SwiGLU)

    def __init__(
        self,
        vocab_size: int = VOCAB_SIZE,
        hidden_size: int = HIDDEN_SIZE,
        num_hidden_layers: int = NUM_HIDDEN_LAYERS,
        num_attention_heads: int = NUM_ATTENTION_HEADS,
        num_key_value_heads: int = NUM_KEY_VALUE_HEADS,
        intermediate_size: int = INTERMEDIATE_SIZE,
        max_position_embeddings: int = MAX_POSITION_EMBEDDINGS,
        rope_theta: float = ROPE_THETA,
        rms_norm_eps: float = RMS_NORM_EPS,
        include_vision: bool = True,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.image_token_id = 151655  # <|image_pad|> token ID

        # Token embeddings
        self.embed_tokens = nn.Embedding(vocab_size, hidden_size)

        # Transformer layers
        self.layers = [
            Qwen3VL2BEncoderLayer(
                hidden_size=hidden_size,
                num_attention_heads=num_attention_heads,
                num_key_value_heads=num_key_value_heads,
                intermediate_size=intermediate_size,
                rms_norm_eps=rms_norm_eps,
                max_position_embeddings=max_position_embeddings,
                rope_theta=rope_theta,
            )
            for _ in range(num_hidden_layers)
        ]

        # Final layer norm
        self.norm = QwenRMSNorm(hidden_size, eps=rms_norm_eps)

        # Rotary embeddings
        self.rotary_emb = QwenRotaryEmbedding(
            dim=hidden_size // num_attention_heads,
            max_position_embeddings=max_position_embeddings,
            base=rope_theta,
            rope_type="default",
        )

        # Vision transformer (optional, loaded separately for embeddings)
        self.visual = None
        self._include_vision = include_vision

    def init_vision(self):
        """Initialize the vision transformer.

        Call this after loading weights if vision is needed.
        This also loads stored vision weights if available.
        """
        if self.visual is None and self._include_vision:
            self.visual = Qwen3VL2BVisionTransformer(
                patch_size=16,
                temporal_patch_size=2,
                in_channels=3,
                embed_dim=self.VISION_EMBED_DIM,
                depth=self.VISION_DEPTH,
                num_heads=self.VISION_NUM_HEADS,
                mlp_ratio=self.VISION_MLP_RATIO,
                hidden_size=self.hidden_size,
                spatial_merge_size=2,
            )

            # Load stored vision weights if available
            if hasattr(self, "_vision_weights") and self._vision_weights:
                from .weights.embedding_weight_handler import load_vision_weights_to_encoder

                load_vision_weights_to_encoder(self)

    def get_image_features(
        self,
        pixel_values: mx.array,
        image_grid_thw: mx.array,
    ) -> list[mx.array]:
        """Extract image features from the vision transformer.

        Args:
            pixel_values: Preprocessed image tensor
            image_grid_thw: Grid dimensions for each image (T, H, W)

        Returns:
            List of image embeddings for each image
        """
        if self.visual is None:
            raise RuntimeError("Vision transformer not initialized. Call init_vision() first.")

        pixel_values = pixel_values.astype(mx.float32)
        image_embeds = self.visual(pixel_values, image_grid_thw)

        # Split embeddings by image
        original_split_sizes = image_grid_thw.prod(axis=-1).astype(mx.int32)

        if bool(mx.any(original_split_sizes == 0)):
            raise ValueError("Invalid image_grid_thw: contains zero-area grids")

        split_sizes = (original_split_sizes // 4).astype(mx.int32)
        split_sizes = [int(s) for s in split_sizes.tolist()]
        split_sizes = [s for s in split_sizes if s > 0]

        if len(split_sizes) == 0:
            raise ValueError("All image grids too small (< 4 tokens each)")

        total_expected = sum(split_sizes)
        if total_expected != image_embeds.shape[0]:
            raise ValueError(
                f"Split size mismatch: expected {total_expected} tokens, "
                f"got {image_embeds.shape[0]} from image embeddings"
            )

        image_embeds_split = []
        start_idx = 0
        for split_size in split_sizes:
            end_idx = start_idx + split_size
            image_embeds_split.append(image_embeds[start_idx:end_idx])
            start_idx = end_idx

        return image_embeds_split

    def __call__(
        self,
        input_ids: mx.array,
        attention_mask: mx.array,
        pixel_values: mx.array | None = None,
        image_grid_thw: mx.array | None = None,
        use_causal_mask: bool = True,
    ) -> mx.array:
        """Forward pass through the encoder.

        Args:
            input_ids: Token IDs [batch, seq_len]
            attention_mask: Attention mask [batch, seq_len]
            pixel_values: Optional image pixels
            image_grid_thw: Optional image grid dimensions
            use_causal_mask: Whether to use causal masking. Set to False for
                embedding mode to enable bidirectional attention (15-30ms speedup).
                Default True for compatibility with autoregressive generation.

        Returns:
            Hidden states [batch, seq_len, hidden_size]
        """
        batch_size, seq_len = input_ids.shape

        # Get token embeddings
        inputs_embeds = self.embed_tokens(input_ids)

        # Insert image embeddings if present
        if pixel_values is not None and image_grid_thw is not None:
            image_embeds_split = self.get_image_features(pixel_values, image_grid_thw)
            image_embeds = mx.concatenate(image_embeds_split, axis=0)

            image_positions = input_ids == self.image_token_id
            n_image_tokens = mx.sum(image_positions).item()

            if n_image_tokens > 0:
                if image_embeds.shape[0] == 0:
                    raise ValueError("Expected image embeddings but got empty array")

                if image_embeds.shape[0] != n_image_tokens:
                    raise ValueError(
                        f"Image token count mismatch: tokenizer found {n_image_tokens} image tokens, "
                        f"but vision model produced {image_embeds.shape[0]} embeddings"
                    )

                if image_embeds.shape[1] != inputs_embeds.shape[-1]:
                    raise ValueError(
                        f"Image embedding dim {image_embeds.shape[1]} != text hidden dim {inputs_embeds.shape[-1]}"
                    )

                # Vectorized image embedding insertion
                image_positions_flat = image_positions.flatten()
                inputs_embeds_flat = inputs_embeds.reshape(-1, inputs_embeds.shape[-1])

                image_indices = mx.cumsum(image_positions_flat.astype(mx.int32), axis=0) - 1
                max_index = max(0, image_embeds.shape[0] - 1)
                image_indices = mx.clip(image_indices, 0, max_index)

                gathered_image_embeds = image_embeds[image_indices]

                new_embeds_flat = mx.where(
                    image_positions_flat[:, None],
                    gathered_image_embeds,
                    inputs_embeds_flat,
                )

                inputs_embeds = new_embeds_flat.reshape(inputs_embeds.shape)

        # Build position embeddings
        cache_position = mx.arange(seq_len, dtype=mx.int32)
        position_ids = mx.expand_dims(mx.expand_dims(cache_position, axis=0), axis=0)
        position_ids = mx.broadcast_to(position_ids, (3, batch_size, seq_len))

        # Build attention mask
        padding_mask = mx.where(
            attention_mask == 1,
            mx.zeros_like(attention_mask).astype(mx.float32),
            mx.ones_like(attention_mask).astype(mx.float32) * (-float("inf")),
        )
        padding_mask = mx.expand_dims(mx.expand_dims(padding_mask, axis=1), axis=1)

        # Build 4D attention mask
        if use_causal_mask:
            # Causal mask for autoregressive generation
            idx = mx.arange(seq_len, dtype=mx.int32)
            j = mx.expand_dims(idx, axis=0)
            i = mx.expand_dims(idx, axis=1)
            tri_bool = j > i
            zeros_2d = mx.zeros((seq_len, seq_len)).astype(mx.float32)
            neginf_2d = mx.ones((seq_len, seq_len)).astype(mx.float32) * (-float("inf"))
            causal_tri_mask = mx.where(tri_bool, neginf_2d, zeros_2d)
            causal_tri_mask = mx.expand_dims(mx.expand_dims(causal_tri_mask, axis=0), axis=0)
            causal_tri_mask = mx.broadcast_to(causal_tri_mask, (batch_size, 1, seq_len, seq_len))
            attention_mask_4d = causal_tri_mask + padding_mask
        else:
            # Bidirectional attention for embedding mode (faster, no causal constraint)
            # Only use padding mask - allows full context for better embeddings
            attention_mask_4d = mx.broadcast_to(padding_mask, (batch_size, 1, seq_len, seq_len))

        # Get rotary embeddings
        hidden_states = inputs_embeds
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        # Forward through transformer layers
        for layer in self.layers:
            hidden_states = layer(hidden_states, attention_mask_4d, position_embeddings)

        # Final layer norm
        hidden_states = self.norm(hidden_states)

        return hidden_states
