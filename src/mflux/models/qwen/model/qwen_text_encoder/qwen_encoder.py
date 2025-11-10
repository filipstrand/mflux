import mlx.core as mx
from mlx import nn

from mflux.models.qwen.model.qwen_text_encoder.qwen_encoder_layer import QwenEncoderLayer
from mflux.models.qwen.model.qwen_text_encoder.qwen_rms_norm import QwenRMSNorm
from mflux.models.qwen.model.qwen_text_encoder.qwen_rope import QwenRotaryEmbedding


class QwenEncoder(nn.Module):
    def __init__(
        self,
        vocab_size: int = 152064,
        hidden_size: int = 3584,
        num_hidden_layers: int = 28,
        max_position_embeddings: int = 128000,
        rope_theta: float = 1000000.0,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.image_token_id = 151655  # <|image_pad|> token

        self.embed_tokens = nn.Embedding(vocab_size, hidden_size)
        self.layers = []
        for i in range(num_hidden_layers):
            layer = QwenEncoderLayer()
            layer._layer_idx = i  # Set layer index for debugging
            layer.self_attn._layer_idx = i  # Set layer index on attention module too
            self.layers.append(layer)
        self.norm = QwenRMSNorm(hidden_size, eps=1e-6)
        self.rotary_emb = QwenRotaryEmbedding(
            dim=hidden_size // 28,
            max_position_embeddings=max_position_embeddings,
            base=rope_theta,
            rope_type="default",
        )

        # Vision transformer for VL fusion (only for Edit model, not txt2img)
        # Will be initialized when visual weights are loaded
        self.visual = None

    def get_image_features(self, pixel_values: mx.array, image_grid_thw: mx.array) -> mx.array:
        """
        Encodes images into continuous embeddings that can be forwarded to the language model.
        Matches the reference implementation exactly.
        """
        if self.visual is None:
            raise RuntimeError("Vision transformer not initialized. Call load_visual_weights() first.")

        # pixel_values can be:
        # - Flattened: [num_patches, C*T*H*W] (from HF processor)
        # - 5D NCDHW: [num_patches, C, T, H, W] (from native patches)
        # Flatten 5D to match PyTorch's approach - patch_embed will reshape
        if len(pixel_values.shape) == 5:
            num_patches = pixel_values.shape[0]
            pixel_values = pixel_values.reshape(num_patches, -1)

        pixel_values = pixel_values.astype(mx.float32)
        image_embeds = self.visual(pixel_values, image_grid_thw)

        # Split embeddings based on grid sizes
        # After patch merger, each image's embeddings are reduced by factor of 4
        # Original grid: H*W patches per image
        # After merger: (H*W)/4 embeddings per image (spatial merge factor = 2x2 = 4)
        # So we need to divide the original grid size by 4 to get the actual split sizes
        original_split_sizes = image_grid_thw.prod(axis=-1).astype(mx.int32)
        # Account for patch merger: divide by 4 (2x2 spatial merge)
        split_sizes = (original_split_sizes // 4).astype(mx.int32)
        split_sizes = [int(s) for s in split_sizes.tolist()]
        # Filter out zero sizes (in case of empty images)
        split_sizes = [s for s in split_sizes if s > 0]

        # Manually split to ensure correct behavior (mx.split may behave differently than torch.split)
        # PyTorch's torch.split(tensor, [196, 196, 196], dim=0) creates 3 tensors of size 196 each
        # We'll manually slice to match this behavior exactly
        image_embeds_split = []
        start_idx = 0
        for split_size in split_sizes:
            end_idx = start_idx + split_size
            if end_idx <= image_embeds.shape[0]:
                image_embeds_split.append(image_embeds[start_idx:end_idx])
                start_idx = end_idx
            else:
                # If we run out of embeddings, break (shouldn't happen if sizes are correct)
                break

        return image_embeds_split

    def get_placeholder_mask(
        self,
        input_ids: mx.array,
        inputs_embeds: mx.array,
        image_features: mx.array = None,
    ) -> mx.array:
        """
        Obtains multimodal placeholder mask from input_ids and checks that the placeholder token count
        is equal to the length of multimodal features. Matches the reference implementation.
        """
        # Find image token positions
        special_image_mask = input_ids == self.image_token_id
        n_image_tokens = mx.sum(special_image_mask).item()

        # Expand mask to match embeddings shape
        special_image_mask = mx.expand_dims(special_image_mask, axis=-1)  # [batch, seq, 1]
        special_image_mask = mx.broadcast_to(special_image_mask, inputs_embeds.shape)  # [batch, seq, hidden]

        # Validate token count matches features
        if image_features is not None:
            total_image_features = image_features.shape[0]  # image_features is already concatenated
            if n_image_tokens != total_image_features:
                raise ValueError(
                    f"Image features and image tokens do not match: tokens: {n_image_tokens}, features {total_image_features}"
                )

        return special_image_mask

    def __call__(
        self,
        input_ids: mx.array | None = None,
        attention_mask: mx.array | None = None,
        inputs_embeds: mx.array | None = None,
        pixel_values: mx.array | None = None,
        image_grid_thw: mx.array | None = None,
        precomputed_image_embeds: mx.array | None = None,
    ) -> mx.array:
        if inputs_embeds is None:
            if input_ids is None:
                raise ValueError("Either input_ids or inputs_embeds must be provided.")
            batch_size, seq_len = input_ids.shape
            inputs_embeds = self.embed_tokens(input_ids)

        else:
            batch_size, seq_len, _ = inputs_embeds.shape

        # Vision-Language fusion - matches reference implementation exactly
        if precomputed_image_embeds is not None and image_grid_thw is not None:
            # Use precomputed embeddings directly - bypass entire vision pipeline
            image_embeds = precomputed_image_embeds

        elif pixel_values is not None and image_grid_thw is not None:
            # Detect format: if pixel_values has 5 dimensions, it's native patches [num_patches, C, T, H, W]
            # If it has 2 dimensions, it's processed embeddings from tokenizer that we need to reject
            if len(pixel_values.shape) == 5:
                # Native format: process through vision transformer
                image_embeds = self.get_image_features(pixel_values, image_grid_thw)
                image_embeds = mx.concatenate(image_embeds, axis=0)
            elif len(pixel_values.shape) == 2:
                # HF processor gives us (num_patches, 1176) where 1176 = C*T*H*W = 3*2*14*14
                # Keep it flattened - patch_embed will reshape it (matching PyTorch's approach)
                # Process through vision transformer - patch_embed expects flattened input
                image_embeds_split = self.get_image_features(pixel_values, image_grid_thw)
                # Concatenate embeddings in order: [image1_embeds, image2_embeds, ...]
                # This order must match the order of images in the prompt
                image_embeds = mx.concatenate(image_embeds_split, axis=0)
            else:
                raise ValueError(f"Unsupported pixel_values format: {pixel_values.shape}")

            # Replace image placeholder tokens with actual image embeddings
            image_positions = input_ids == self.image_token_id
            n_image_tokens = mx.sum(image_positions).item()

            if n_image_tokens > 0 and image_embeds.shape[0] >= n_image_tokens:
                # Replace image tokens with image embeddings
                # Create a list of embeddings and then stack them
                image_positions_flat = image_positions.flatten()
                inputs_embeds_flat = inputs_embeds.reshape(-1, inputs_embeds.shape[-1])

                # Match PyTorch exactly: concatenate all embeddings and use sequentially
                # PyTorch does: image_embeds = torch.cat(image_embeds_split, dim=0) then masked_scatter
                # masked_scatter fills positions sequentially from the source tensor
                # PyTorch's get_placeholder_mask validates that tokens == embeddings, so they must match
                # For multi-image, PyTorch produces more tokens to match all embeddings
                # We should use all available embeddings if we have enough tokens, otherwise truncate
                if image_embeds.shape[0] <= n_image_tokens:
                    # We have enough tokens - use all embeddings (matches PyTorch)
                    image_embeds_to_use = image_embeds
                    if len(image_embeds_split) > 1:
                        print(
                            f"🔎 QwenEncoder: Multi-image - using all {image_embeds.shape[0]} embeddings with {n_image_tokens} tokens"
                        )
                else:
                    # More embeddings than tokens - truncate to match token count
                    # This should only happen if tokenization was incorrect
                    image_embeds_to_use = image_embeds[:n_image_tokens]
                    print(
                        f"🔎 QwenEncoder: WARNING - More embeddings ({image_embeds.shape[0]}) than tokens ({n_image_tokens}), truncating embeddings"
                    )

                if len(image_embeds_split) > 1:
                    split_sizes = [e.shape[0] for e in image_embeds_split]
                    if image_embeds.shape[0] <= n_image_tokens:
                        print(
                            f"🔎 QwenEncoder: Multi-image: Have {split_sizes} embeddings per image, using all {image_embeds.shape[0]} embeddings"
                        )
                    else:
                        print(
                            f"🔎 QwenEncoder: Multi-image: Have {split_sizes} embeddings per image, using first {n_image_tokens} of {image_embeds.shape[0]} total"
                        )

                    # Calculate which embeddings from which split are being used
                    # If we have [196, 196] splits and use 237 tokens:
                    # - First 196 come from split[0] (image1)
                    # - Next 41 come from split[1] (image2)
                    split_usage = []
                    remaining = n_image_tokens
                    for i, split_size in enumerate(split_sizes):
                        used_from_split = min(split_size, remaining)
                        split_usage.append(
                            {
                                "split_idx": i,
                                "used": used_from_split,
                                "total": split_size,
                                "start_idx_in_concat": sum(split_sizes[:i]),
                                "end_idx_in_concat": sum(split_sizes[:i]) + used_from_split,
                            }
                        )
                        remaining -= used_from_split
                        if remaining <= 0:
                            break

                # Build new embeddings list
                new_embeds_list = []
                image_idx = 0
                for i in range(len(image_positions_flat)):
                    if image_positions_flat[i] and image_idx < image_embeds_to_use.shape[0]:
                        new_embeds_list.append(image_embeds_to_use[image_idx])
                        image_idx += 1
                    else:
                        new_embeds_list.append(inputs_embeds_flat[i])

                # Stack the new embeddings
                new_embeds = mx.stack(new_embeds_list, axis=0)
                inputs_embeds = new_embeds.reshape(inputs_embeds.shape)

        cache_position = mx.arange(seq_len, dtype=mx.int32)
        position_ids = mx.expand_dims(mx.expand_dims(cache_position, axis=0), axis=0)
        position_ids = mx.broadcast_to(position_ids, (3, batch_size, seq_len))
        if attention_mask is not None:
            padding_mask = mx.where(
                attention_mask == 1,
                mx.zeros_like(attention_mask).astype(mx.float32),
                mx.ones_like(attention_mask).astype(mx.float32) * (-float("inf")),
            )
            padding_mask = mx.expand_dims(mx.expand_dims(padding_mask, axis=1), axis=1)
        else:
            padding_mask = None

        # Create causal triangular mask
        idx = mx.arange(seq_len, dtype=mx.int32)
        j = mx.expand_dims(idx, axis=0)
        i = mx.expand_dims(idx, axis=1)
        tri_bool = j > i
        zeros_2d = mx.zeros((seq_len, seq_len)).astype(mx.float32)
        neginf_2d = mx.ones((seq_len, seq_len)).astype(mx.float32) * (-float("inf"))
        causal_tri_mask = mx.where(tri_bool, neginf_2d, zeros_2d)
        causal_tri_mask = mx.expand_dims(mx.expand_dims(causal_tri_mask, axis=0), axis=0)  # [1,1,S,S]
        causal_tri_mask = mx.broadcast_to(causal_tri_mask, (batch_size, 1, seq_len, seq_len))

        # Final attention mask: causal triangle + padding mask
        if padding_mask is not None:
            attention_mask_4d = causal_tri_mask + padding_mask
        else:
            attention_mask_4d = causal_tri_mask

        hidden_states = inputs_embeds
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        # Transformer layers
        for i, layer in enumerate(self.layers):
            hidden_states = layer(hidden_states, attention_mask_4d, position_embeddings)

        # Apply norm AFTER ALL layers (matching PyTorch reference)
        hidden_states = self.norm(hidden_states)

        return hidden_states
