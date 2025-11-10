import mlx.core as mx
from mlx import nn

from mflux.models.qwen.model.qwen_text_encoder.qwen_encoder import QwenEncoder


class QwenTextEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = QwenEncoder()

    def __call__(
        self,
        input_ids: mx.array,
        attention_mask: mx.array,
    ) -> tuple[mx.array, mx.array]:
        hidden_states = self.encoder(input_ids, attention_mask)

        prompt_embeds, encoder_attention_mask = QwenTextEncoder._process_text_embeddings_mlx(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            drop_idx=34,
            dtype=mx.bfloat16,
        )

        return prompt_embeds, encoder_attention_mask

    @staticmethod
    def _process_text_embeddings_mlx(hidden_states, attention_mask, drop_idx=1, dtype=mx.float32):
        split_hidden_states = QwenTextEncoder._extract_masked_hidden(hidden_states, attention_mask)
        split_hidden_states = [e[drop_idx:] for e in split_hidden_states]
        attn_mask_list = [mx.ones(e.shape[0], dtype=mx.int32) for e in split_hidden_states]
        max_seq_len = max([e.shape[0] for e in split_hidden_states])

        padded_embeds = []
        for u in split_hidden_states:
            current_len = u.shape[0]
            hidden_dim = u.shape[1]
            if current_len < max_seq_len:
                padding = mx.zeros((max_seq_len - current_len, hidden_dim), dtype=u.dtype)
                padded = mx.concatenate([u, padding], axis=0)
            else:
                padded = u
            padded_embeds.append(padded)

        prompt_embeds = mx.stack(padded_embeds, axis=0)

        padded_masks = []
        for mask in attn_mask_list:
            current_len = mask.shape[0]
            if current_len < max_seq_len:
                padding = mx.zeros(max_seq_len - current_len, dtype=mask.dtype)
                padded = mx.concatenate([mask, padding], axis=0)
            else:
                padded = mask
            padded_masks.append(padded)

        encoder_attention_mask = mx.stack(padded_masks, axis=0)
        prompt_embeds = prompt_embeds.astype(dtype)
        return prompt_embeds, encoder_attention_mask

    @staticmethod
    def _extract_masked_hidden(hidden_states, attention_mask):
        batch_size = hidden_states.shape[0]
        split_hidden_states = []
        for i in range(batch_size):
            mask = attention_mask[i]
            valid_length = mx.sum(mask).item()
            valid_length = int(valid_length)
            valid_hidden = hidden_states[i, :valid_length, :]
            split_hidden_states.append(valid_hidden)
        return split_hidden_states
