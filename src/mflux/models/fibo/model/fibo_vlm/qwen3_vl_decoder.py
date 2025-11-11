"""
Qwen3VL Decoder for FIBO VLM.

Custom implementation using only FIBO VLM-specific components.
No dependencies on Qwen components.
"""

import mlx.core as mx
from mlx import nn

from .qwen3_vl_decoder_layer import Qwen3VLDecoderLayer
from .qwen3_vl_rms_norm import Qwen3VLRMSNorm
from .qwen3_vl_rope import Qwen3VLRotaryEmbedding


class Qwen3VLDecoder(nn.Module):
    def __init__(
        self,
        vocab_size: int = 151936,
        hidden_size: int = 2560,
        num_hidden_layers: int = 36,
        num_attention_heads: int = 32,
        num_key_value_heads: int = 8,
        intermediate_size: int = 9728,
        max_position_embeddings: int = 262144,
        rope_theta: float = 5000000.0,
        rms_norm_eps: float = 1e-6,
        head_dim: int | None = None,
        attention_bias: bool = False,
        mrope_section: list[int] | None = None,
        attention_scaling: float = 1.0,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers

        # Qwen3VL uses explicit head_dim (defaults to 128 if not provided)
        if head_dim is None:
            head_dim = 128  # Default for Qwen3VL

        if mrope_section is None:
            # Fallback to Qwen default for text if not provided
            mrope_section = [24, 20, 20]

        self.embed_tokens = nn.Embedding(vocab_size, hidden_size)

        # Create decoder layers with custom FIBO VLM components
        self.layers = [
            Qwen3VLDecoderLayer(
                hidden_size=hidden_size,
                num_attention_heads=num_attention_heads,
                num_key_value_heads=num_key_value_heads,
                head_dim=head_dim,
                max_position_embeddings=max_position_embeddings,
                rope_theta=rope_theta,
                mrope_section=mrope_section,
                attention_bias=attention_bias,
                rms_norm_eps=rms_norm_eps,
                intermediate_size=intermediate_size,
            )
            for _ in range(num_hidden_layers)
        ]
        self.norm = Qwen3VLRMSNorm(hidden_size, eps=rms_norm_eps)
        self.rotary_emb = Qwen3VLRotaryEmbedding(
            dim=head_dim,
            max_position_embeddings=max_position_embeddings,
            base=rope_theta,
            scaling_factor=attention_scaling,
            mrope_section=mrope_section,
        )

        # Language modeling head for generation
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)

        # Vision components (for multimodal support) - will be initialized separately
        self.visual = None
        self.image_token_id = None  # Will be set from config

    def __call__(
        self,
        input_ids: mx.array,
        attention_mask: mx.array | None = None,
        pixel_values: mx.array | None = None,
        image_grid_thw: mx.array | None = None,
        use_cache: bool = False,
        past_key_values: list[tuple[mx.array, mx.array]] | None = None,
    ) -> mx.array | tuple[mx.array, list[tuple[mx.array, mx.array]]]:
        batch_size, seq_len = input_ids.shape

        # Get embeddings
        inputs_embeds = self.embed_tokens(input_ids)

        # Handle vision inputs if provided
        if pixel_values is not None and image_grid_thw is not None and self.visual is not None:
            # Process vision inputs (similar to QwenEncoder)
            image_embeds_split = self._get_image_features(pixel_values, image_grid_thw)
            image_embeds = mx.concatenate(image_embeds_split, axis=0)

            if self.image_token_id is not None:
                image_positions = input_ids == self.image_token_id
                n_image_tokens = mx.sum(image_positions).item()

                if n_image_tokens > 0 and image_embeds.shape[0] >= n_image_tokens:
                    image_positions_flat = image_positions.flatten()
                    inputs_embeds_flat = inputs_embeds.reshape(-1, inputs_embeds.shape[-1])

                    new_embeds_list = []
                    image_idx = 0
                    for i in range(len(image_positions_flat)):
                        if image_positions_flat[i] and image_idx < image_embeds.shape[0]:
                            new_embeds_list.append(image_embeds[image_idx])
                            image_idx += 1
                        else:
                            new_embeds_list.append(inputs_embeds_flat[i])

                    new_embeds = mx.stack(new_embeds_list, axis=0)
                    inputs_embeds = new_embeds.reshape(inputs_embeds.shape)

        # Create attention mask with causal masking (from QwenEncoder)
        if attention_mask is None:
            attention_mask = mx.ones((batch_size, seq_len), dtype=mx.int32)

        # Position embeddings
        # If using cache, compute position IDs relative to cached sequence length
        if use_cache and past_key_values is not None:
            # Get cached sequence length from first layer's key cache
            cached_seq_len = past_key_values[0][0].shape[2] if len(past_key_values) > 0 else 0
            # Position IDs for new tokens start from cached_seq_len
            cache_position = mx.arange(cached_seq_len, cached_seq_len + seq_len, dtype=mx.int32)
        else:
            # First iteration: positions start from 0
            cache_position = mx.arange(seq_len, dtype=mx.int32)
        position_ids = mx.expand_dims(mx.expand_dims(cache_position, axis=0), axis=0)
        position_ids = mx.broadcast_to(position_ids, (3, batch_size, seq_len))

        # Create causal + padding mask
        # Use the dtype of inputs_embeds for masks (usually float16)
        mask_dtype = inputs_embeds.dtype
        padding_mask = mx.where(
            attention_mask == 1,
            mx.zeros(attention_mask.shape, dtype=mask_dtype),
            mx.full(attention_mask.shape, -float("inf"), dtype=mask_dtype),
        )
        padding_mask = mx.expand_dims(mx.expand_dims(padding_mask, axis=1), axis=1)

        # Optimize: when seq_len=1 (cached iterations), causal mask is trivial (all zeros)
        if seq_len == 1:
            causal_tri_mask = mx.zeros((batch_size, 1, 1, 1), dtype=mask_dtype)
        else:
            idx = mx.arange(seq_len, dtype=mx.int32)
            j = mx.expand_dims(idx, axis=0)
            i = mx.expand_dims(idx, axis=1)
            tri_bool = j > i
            zeros_2d = mx.zeros((seq_len, seq_len), dtype=mask_dtype)
            neginf_2d = mx.full((seq_len, seq_len), -float("inf"), dtype=mask_dtype)
            causal_tri_mask = mx.where(tri_bool, neginf_2d, zeros_2d)
            causal_tri_mask = mx.expand_dims(mx.expand_dims(causal_tri_mask, axis=0), axis=0)
            causal_tri_mask = mx.broadcast_to(causal_tri_mask, (batch_size, 1, seq_len, seq_len))
        attention_mask_4d = causal_tri_mask + padding_mask

        # Forward through layers
        hidden_states = inputs_embeds
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        # Handle KV cache - use provided past_key_values or None
        # past_key_values is a list of (key, value) tuples, one per layer

        # Forward through layers (decoder layers verified - checkpoints removed)
        present_key_values = [] if use_cache else None
        for layer_idx, layer in enumerate(self.layers):
            layer_past = None
            if use_cache and past_key_values is not None:
                layer_past = past_key_values[layer_idx] if layer_idx < len(past_key_values) else None

            layer_output = layer(
                hidden_states,
                attention_mask_4d,
                position_embeddings,
                past_key_value=layer_past,
            )

            if use_cache:
                hidden_states, layer_present = layer_output
                present_key_values.append(layer_present)
            else:
                hidden_states = layer_output

        # Final norm
        hidden_states = self.norm(hidden_states)

        # Use float32 for lm_head computation to avoid numerical precision issues
        # with large matrix multiplications (151936 vocab size)
        # Cast inputs to float32, compute, then cast back to float16
        hidden_states_f32 = hidden_states.astype(mx.float32)
        weight_f32 = self.lm_head.weight.astype(mx.float32)

        # Perform matrix multiplication in float32
        # MLX Linear does: input @ weight.T
        logits_f32 = mx.matmul(hidden_states_f32, weight_f32.T)

        # Cast back to float16 to match expected output dtype
        logits = logits_f32.astype(hidden_states.dtype)

        if use_cache:
            return logits, present_key_values
        return logits

    def _get_image_features(self, pixel_values: mx.array, image_grid_thw: mx.array) -> list[mx.array]:
        if self.visual is None:
            raise RuntimeError("Vision transformer not initialized. Call load_visual_weights() first.")

        # Vision model can handle the original dtype (usually float16)
        image_embeds, _ = self.visual(pixel_values, image_grid_thw)  # Ignore deepstack for now
        original_split_sizes = image_grid_thw.prod(axis=-1).astype(mx.int32)
        spatial_merge_size = self.visual.spatial_merge_size
        split_sizes = (original_split_sizes // (spatial_merge_size**2)).astype(mx.int32)
        split_sizes = [int(s) for s in split_sizes.tolist()]
        split_sizes = [s for s in split_sizes if s > 0]

        image_embeds_split = []
        start_idx = 0
        for split_size in split_sizes:
            end_idx = start_idx + split_size
            image_embeds_split.append(image_embeds[start_idx:end_idx])
            start_idx = end_idx
        return image_embeds_split
