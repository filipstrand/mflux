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
        self.image_token_id = 151655

        self.embed_tokens = nn.Embedding(vocab_size, hidden_size)
        self.layers = [QwenEncoderLayer() for i in range(num_hidden_layers)]
        self.norm = QwenRMSNorm(hidden_size, eps=1e-6)
        self.rotary_emb = QwenRotaryEmbedding(
            dim=hidden_size // 28,
            max_position_embeddings=max_position_embeddings,
            base=rope_theta,
            rope_type="default",
        )

        self.visual = None

    def get_image_features(self, pixel_values: mx.array, image_grid_thw: mx.array) -> mx.array:
        if self.visual is None:
            raise RuntimeError("Vision transformer not initialized. Call load_visual_weights() first.")

        pixel_values = pixel_values.astype(mx.float32)
        image_embeds = self.visual(pixel_values, image_grid_thw)
        original_split_sizes = image_grid_thw.prod(axis=-1).astype(mx.int32)
        split_sizes = (original_split_sizes // 4).astype(mx.int32)
        split_sizes = [int(s) for s in split_sizes.tolist()]
        split_sizes = [s for s in split_sizes if s > 0]
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
    ) -> mx.array:
        batch_size, seq_len = input_ids.shape
        inputs_embeds = self.embed_tokens(input_ids)

        if pixel_values is not None and image_grid_thw is not None:
            image_embeds_split = self.get_image_features(pixel_values, image_grid_thw)
            image_embeds = mx.concatenate(image_embeds_split, axis=0)

            image_positions = input_ids == self.image_token_id
            n_image_tokens = mx.sum(image_positions).item()

            if n_image_tokens > 0 and image_embeds.shape[0] >= n_image_tokens:
                image_positions_flat = image_positions.flatten()
                inputs_embeds_flat = inputs_embeds.reshape(-1, inputs_embeds.shape[-1])
                image_embeds_to_use = image_embeds

                new_embeds_list = []
                image_idx = 0
                for i in range(len(image_positions_flat)):
                    if image_positions_flat[i] and image_idx < image_embeds_to_use.shape[0]:
                        new_embeds_list.append(image_embeds_to_use[image_idx])
                        image_idx += 1
                    else:
                        new_embeds_list.append(inputs_embeds_flat[i])

                new_embeds = mx.stack(new_embeds_list, axis=0)
                inputs_embeds = new_embeds.reshape(inputs_embeds.shape)
        cache_position = mx.arange(seq_len, dtype=mx.int32)
        position_ids = mx.expand_dims(mx.expand_dims(cache_position, axis=0), axis=0)
        position_ids = mx.broadcast_to(position_ids, (3, batch_size, seq_len))
        padding_mask = mx.where(
            attention_mask == 1,
            mx.zeros_like(attention_mask).astype(mx.float32),
            mx.ones_like(attention_mask).astype(mx.float32) * (-float("inf")),
        )
        padding_mask = mx.expand_dims(mx.expand_dims(padding_mask, axis=1), axis=1)

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
        hidden_states = inputs_embeds
        position_embeddings = self.rotary_emb(hidden_states, position_ids)
        for i, layer in enumerate(self.layers):
            hidden_states = layer(hidden_states, attention_mask_4d, position_embeddings)

        hidden_states = self.norm(hidden_states)
        return hidden_states
