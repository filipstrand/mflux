import mlx.core as mx
from mlx import nn

from mflux.models.qwen.model.qwen_text_encoder.qwen_encoder import QwenEncoder


class QwenVisionLanguageEncoder(nn.Module):
    def __init__(self, encoder=None):
        super().__init__()
        self.encoder = encoder or QwenEncoder()
        self.edit_template_start_idx = 64

    def __call__(
        self,
        input_ids: mx.array,
        attention_mask: mx.array | None = None,
        pixel_values: mx.array | None = None,
        image_grid_thw: mx.array | None = None,
    ) -> tuple[mx.array, mx.array]:
        hidden_states = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
        )

        trimmed = []
        batch = hidden_states.shape[0]
        for i in range(batch):
            valid_len = int(mx.sum(attention_mask[i]).item())
            trimmed.append(hidden_states[i, :valid_len, :])
        drop_idx = self.edit_template_start_idx
        trimmed_after_drop = [t[drop_idx:] if t.shape[0] > drop_idx else t for t in trimmed]
        trimmed = trimmed_after_drop
        max_len = max(t.shape[0] for t in trimmed) if trimmed else 0
        hidden_dim = hidden_states.shape[2]
        padded_embeds = []
        padded_masks = []
        for t in trimmed:
            cur_len = t.shape[0]
            if cur_len < max_len:
                pad_e = mx.zeros((max_len - cur_len, hidden_dim), dtype=t.dtype)
                t_pad = mx.concatenate([t, pad_e], axis=0)
                pad_m = mx.concatenate(
                    [mx.ones(cur_len, dtype=mx.int32), mx.zeros(max_len - cur_len, dtype=mx.int32)], axis=0
                )
            else:
                t_pad = t
                pad_m = mx.ones(cur_len, dtype=mx.int32)
            padded_embeds.append(t_pad)
            padded_masks.append(pad_m)
        prompt_embeds = mx.stack(padded_embeds, axis=0) if padded_embeds else hidden_states
        encoder_attention_mask = mx.stack(padded_masks, axis=0) if padded_masks else attention_mask
        return prompt_embeds, encoder_attention_mask
