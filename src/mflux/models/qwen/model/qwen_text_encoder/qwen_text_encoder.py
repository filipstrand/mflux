import mlx.core as mx
from mlx import nn

from mflux.models.qwen.model.qwen_text_encoder.qwen_encoder import QwenEncoder


class QwenTextEncoder(nn.Module):
    def __init__(
        self,
        vocab_size: int = 152064,
        hidden_size: int = 3584,
        num_hidden_layers: int = 28,
        num_attention_heads: int = 28,
        num_key_value_heads: int = 4,
        intermediate_size: int = 18944,
        max_position_embeddings: int = 128000,
        rms_norm_eps: float = 1e-6,
    ):
        super().__init__()

        self.encoder = QwenEncoder(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=num_attention_heads,
            num_key_value_heads=num_key_value_heads,
            intermediate_size=intermediate_size,
            max_position_embeddings=max_position_embeddings,
            rms_norm_eps=rms_norm_eps,
            rope_theta=1000000.0,
            rope_scaling={"mrope_section": [16, 24, 24]},
        )

    def __call__(
        self,
        input_ids: mx.array,
        attention_mask: mx.array | None = None,
        position_ids: mx.array | None = None,
    ) -> mx.array:
        hidden_states = self.encoder(input_ids, attention_mask)
        input_seq_len = input_ids.shape[1]

        # Extract valid hidden states from the single batch item
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