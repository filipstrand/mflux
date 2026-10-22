import inspect

import mlx.core as mx
import numpy as np
from mlx import nn

from mflux.models.common_models.qwen3_vl.qwen3_vl_vision_model import Qwen3VLVisionModel
from mflux.models.qwen21.model.qwen21_text_encoder.language_model import LanguageModel


class QwenImage21TextEncoder(nn.Module):
    def __init__(self, config: dict):
        super().__init__()
        self.language_model = LanguageModel(config["text_config"])
        vision = {
            key: value
            for key, value in config["vision_config"].items()
            if key in inspect.signature(Qwen3VLVisionModel).parameters
        }
        self.visual = Qwen3VLVisionModel(**vision, preserve_input_dtype=True)
        for block in self.visual.blocks:
            block.mlp.act_fn = self._tanh_gelu
        for merger in [self.visual.merger, *self.visual.deepstack_merger_list]:
            merger.act_fn = self._exact_gelu
        text = config["text_config"]
        self.lm_head = nn.Linear(text["hidden_size"], text["vocab_size"], bias=False)
        self.has_generation_head = True  # the initializer clears it for checkpoints without lm_head
        self.image_token_id = config["image_token_id"]
        self.merge_size = config["vision_config"]["spatial_merge_size"]

    def __call__(
        self, input_ids: mx.array, pixel_values: mx.array | None = None, image_grid_thw: mx.array | None = None
    ) -> mx.array:
        if input_ids.shape[0] != 1:
            raise ValueError("Qwen-Image-2.1 currently encodes one prompt at a time.")
        hidden = self.language_model.embed_tokens(input_ids)
        image_indices = mx.array(np.flatnonzero(np.asarray(input_ids[0]) == self.image_token_id), dtype=mx.int32)
        deepstack = None
        if pixel_values is not None:
            features, deepstack = self.visual(pixel_values.astype(hidden.dtype), image_grid_thw, return_deepstack=True)
            if features.shape[0] != image_indices.size:
                raise ValueError("Vision feature count does not match the image placeholders.")
            hidden[:, image_indices] = features.astype(hidden.dtype)[None]
        positions = self.position_ids(input_ids, image_grid_thw)
        return self.language_model(hidden, positions, image_indices, deepstack)

    def generate(
        self,
        input_ids: mx.array,
        pixel_values: mx.array | None = None,
        image_grid_thw: mx.array | None = None,
        max_new_tokens: int = 64,
        stop_token_ids: tuple[int, ...] = (151645, 151643),
    ) -> list[int]:
        # Greedy decoding with the Qwen3-VL the prompt encoder already holds (its lm_head
        # is untied): one vision+prompt prefill into per-layer KV caches, then single-token
        # steps. Deepstack injects during the prefill exactly as in __call__.
        if not self.has_generation_head:
            raise RuntimeError("This checkpoint has no text-encoder lm_head; re-save it from the official weights.")
        model = self.language_model
        hidden = model.embed_tokens(input_ids)
        image_indices = mx.array(np.flatnonzero(np.asarray(input_ids[0]) == self.image_token_id), dtype=mx.int32)
        deepstack = None
        if pixel_values is not None:
            features, deepstack = self.visual(pixel_values.astype(hidden.dtype), image_grid_thw, return_deepstack=True)
            if features.shape[0] != image_indices.size:
                raise ValueError("Vision feature count does not match the image placeholders.")
            hidden[:, image_indices] = features.astype(hidden.dtype)[None]
        positions = self.position_ids(input_ids, image_grid_thw)
        rope = model.rotary_emb(hidden, positions)
        idx = mx.arange(hidden.shape[1])
        mask = (idx[:, None] >= idx[None, :])[None, None]
        total = hidden.shape[1] + max_new_tokens
        caches = []
        for index, layer in enumerate(model.layers):
            hidden, cache = layer(hidden, attention_mask=mask, position_embeddings=rope, max_cache_length=total)
            if deepstack is not None and index < len(deepstack):
                hidden[:, image_indices] += deepstack[index].astype(hidden.dtype)[None]
            caches.append(cache)
        mx.eval(hidden)
        position = int(np.asarray(positions).max()) + 1
        generated: list[int] = []
        for _ in range(max_new_tokens):
            next_id = int(mx.argmax(self.lm_head(model.norm(hidden[:, -1]))[0]).item())
            if next_id in stop_token_ids:
                break
            generated.append(next_id)
            hidden = model.embed_tokens(mx.array([[next_id]]))
            rope = model.rotary_emb(hidden, mx.full((3, 1, 1), position, dtype=mx.int32))
            for index, layer in enumerate(model.layers):
                hidden, caches[index] = layer(hidden, position_embeddings=rope, past_key_value=caches[index])
            mx.eval(hidden)
            position += 1
        return generated

    def position_ids(self, input_ids: mx.array, grids: mx.array | None) -> mx.array:
        return QwenImage21TextEncoder.mrope_positions(input_ids, grids, self.image_token_id, self.merge_size)

    @staticmethod
    def mrope_positions(input_ids: mx.array, grids: mx.array | None, image_token_id: int, merge_size: int) -> mx.array:
        ids = np.asarray(input_ids[0])
        result = np.zeros((3, len(ids)), dtype=np.int32)
        cursor, position = 0, 0
        if grids is not None:
            for t, h, w in grids.tolist():
                h, w = h // merge_size, w // merge_size
                start = int(np.flatnonzero(ids[cursor:] == image_token_id)[0]) + cursor
                result[:, cursor:start] = np.arange(position, position + start - cursor)[None]
                position += start - cursor
                length = t * h * w
                result[:, start : start + length] = (
                    np.stack(
                        [
                            np.repeat(np.arange(t), h * w),
                            np.tile(np.repeat(np.arange(h), w), t),
                            np.tile(np.arange(w), t * h),
                        ]
                    )
                    + position
                )
                cursor = start + length
                position += max(h, w)
        result[:, cursor:] = np.arange(position, position + len(ids) - cursor)[None]
        return mx.array(result[:, None])

    @staticmethod
    def _exact_gelu(value: mx.array) -> mx.array:
        return nn.gelu(value.astype(mx.float32)).astype(value.dtype)

    @staticmethod
    def _tanh_gelu(value: mx.array) -> mx.array:
        return nn.gelu_approx(value.astype(mx.float32)).astype(value.dtype)
