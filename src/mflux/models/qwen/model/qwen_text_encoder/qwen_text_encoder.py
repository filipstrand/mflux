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
        attention_mask: mx.array | None = None,
        position_ids: int = 0,
    ) -> [mx.array, mx.array]:
        # Get hidden states from the encoder
        hidden_states = self.encoder(input_ids, attention_mask)

        # Extract valid hidden states
        input_seq_len = input_ids.shape[1]
        valid_hidden = hidden_states[0, :input_seq_len, :]
        if 0 < position_ids < input_seq_len:
            valid_hidden = valid_hidden[position_ids:, :]
            valid_length = input_seq_len - position_ids
        else:
            valid_length = input_seq_len

        # Create mask and add batch dimension back
        encoder_attention_mask = mx.ones((valid_length,), dtype=mx.int64)
        prompt_embeds = mx.expand_dims(valid_hidden, axis=0)  # Add batch dimension
        encoder_attention_mask = mx.expand_dims(encoder_attention_mask, axis=0)  # Add batch dimension

        return prompt_embeds, encoder_attention_mask
