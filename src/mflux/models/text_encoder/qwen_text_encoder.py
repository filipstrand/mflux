import mlx.core as mx
from mlx import nn

from mflux.models.text_encoder.qwen_encoder.qwen_encoder import QwenEncoder


class QwenTextEncoder(nn.Module):
    """
    Qwen Text Encoder for MFLUX.

    This encoder follows the same pattern as T5Encoder and CLIPEncoder
    in the MFLUX codebase, providing a clean interface for text encoding
    in the Qwen image generation pipeline.
    """

    def __init__(
        self,
        vocab_size: int = 152064,  # Actual Qwen2.5-VL vocab size from config
        hidden_size: int = 3584,  # Match joint_attention_dim in transformer
        num_hidden_layers: int = 28,  # From config
        num_attention_heads: int = 28,  # From config
        num_key_value_heads: int = 4,  # GQA: only 4 KV heads from config
        intermediate_size: int = 18944,  # From config
        max_position_embeddings: int = 128000,  # From config
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
            rope_theta=1000000.0,  # From config
            rope_scaling={"mrope_section": [16, 24, 24]},  # Default from config
        )

    def __call__(
        self, input_ids: mx.array, attention_mask: mx.array | None = None, position_ids: mx.array | None = None
    ) -> mx.array:
        """
        Encode text tokens to embeddings.

        Args:
            input_ids: Token IDs of shape [batch_size, seq_len]
            attention_mask: Attention mask of shape [batch_size, seq_len]

        Returns:
            Text embeddings of shape [batch_size, seq_len, 3584]
        """
        return self.encoder(input_ids, attention_mask, position_ids)

    def encode_with_mask_extraction(
        self, input_ids: mx.array, attention_mask: mx.array, template_start_idx: int
    ) -> tuple[mx.array, mx.array]:
        """
        Encode text and extract valid portions based on attention mask.

        This method replicates the logic from the diffusers Qwen pipeline
        where template prefixes are dropped and valid sequences are extracted.

        Args:
            input_ids: Token IDs of shape [batch_size, seq_len]
            attention_mask: Attention mask of shape [batch_size, seq_len]
            template_start_idx: Number of tokens to drop from template prefix

        Returns:
            tuple: (prompt_embeds, encoder_attention_mask)
                - prompt_embeds: Padded embeddings [batch_size, max_valid_len, 3584]
                - encoder_attention_mask: Attention mask [batch_size, max_valid_len]
        """
        # Encode the full sequence
        hidden_states = self.encoder(input_ids, attention_mask)

        # Extract valid sequences based on attention mask
        batch_size = hidden_states.shape[0]
        valid_sequences = []
        valid_masks = []

        for i in range(batch_size):
            # Find valid tokens (where attention_mask is 1)
            mask = attention_mask[i]
            valid_length = int(mx.sum(mask).item())

            # Extract valid hidden states by slicing instead of boolean indexing
            if valid_length > 0:
                valid_hidden = hidden_states[i, :valid_length, :]  # [valid_length, hidden_size]
                if template_start_idx > 0 and valid_length > template_start_idx:
                    valid_hidden = valid_hidden[template_start_idx:, :]  # Drop template prefix
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
