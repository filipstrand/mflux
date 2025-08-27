import mlx.core as mx
from mlx import nn

from mflux.models.qwen_text_encoder.qwen_encoder import QwenEncoder


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
        return self.encoder(input_ids, attention_mask, position_ids)

    def encode(
        self,
        input_ids: mx.array,
        attention_mask: mx.array,
        template_start_idx: int,
    ) -> tuple[mx.array, mx.array]:
        hidden_states = self.encoder(input_ids, attention_mask)
        batch_size = hidden_states.shape[0]
        valid_sequences = []
        valid_masks = []

        for i in range(batch_size):
            input_seq_len = input_ids.shape[1]
            valid_length = input_seq_len

            # Extract valid hidden states by slicing instead of boolean indexing
            if valid_length > 0:
                valid_hidden = hidden_states[i, :valid_length, :]
                if 0 < template_start_idx < valid_length:
                    valid_hidden = valid_hidden[template_start_idx:, :]
                    valid_length = valid_length - template_start_idx
            else:
                valid_hidden = mx.zeros((1, hidden_states.shape[-1]), dtype=hidden_states.dtype)
                valid_length = 1

            valid_sequences.append(valid_hidden)
            valid_masks.append(mx.ones((valid_length,), dtype=mx.int64))

        # Pad sequences to same length
        if valid_sequences:
            max_length = max(seq.shape[0] for seq in valid_sequences)

            padded_sequences = []
            padded_masks = []

            for seq, mask in zip(valid_sequences, valid_masks):
                seq_len = seq.shape[0]
                if seq_len < max_length:
                    # Pad with zeros
                    padding = mx.zeros((max_length - seq_len, seq.shape[1]), dtype=seq.dtype)
                    seq = mx.concatenate([seq, padding], axis=0)

                    mask_padding = mx.zeros((max_length - seq_len,), dtype=mask.dtype)
                    mask = mx.concatenate([mask, mask_padding], axis=0)

                padded_sequences.append(seq)
                padded_masks.append(mask)

            prompt_embeds = mx.stack(padded_sequences, axis=0)
            encoder_attention_mask = mx.stack(padded_masks, axis=0)
        else:
            # Handle empty case
            prompt_embeds = mx.zeros((batch_size, 1, hidden_states.shape[-1]), dtype=hidden_states.dtype)
            encoder_attention_mask = mx.zeros((batch_size, 1), dtype=mx.int64)

        return prompt_embeds, encoder_attention_mask
