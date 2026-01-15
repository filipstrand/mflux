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
        # MEDIUM FIX: drop_idx=34 for Qwen model (drops first 34 tokens per model spec)
        # HIGH FIX: Validate sequences have sufficient length before dropping tokens
        split_hidden_states = QwenTextEncoder._extract_masked_hidden(hidden_states, attention_mask)
        split_hidden_states = [e[drop_idx:] for e in split_hidden_states]

        # HIGH FIX: Check for empty sequences after drop_idx (e.g., all sequences ≤ drop_idx tokens)
        if not split_hidden_states or all(e.shape[0] == 0 for e in split_hidden_states):
            raise ValueError(
                f"All sequences too short (≤{drop_idx} tokens) after masking. "
                f"Qwen text encoder requires sequences with >{drop_idx} tokens."
            )

        attn_mask_list = [mx.ones(e.shape[0], dtype=mx.int32) for e in split_hidden_states]
        # MEDIUM FIX: Use generator expression instead of list comprehension for efficiency
        max_seq_len = max(e.shape[0] for e in split_hidden_states)

        # OPTIMIZATION: Vectorized padding with mx.pad() instead of zeros + concatenate
        # Old: Python loop with dynamic allocation (mx.zeros) + concatenate per element
        # New: Use mx.pad() which is optimized for this exact use case
        padded_embeds = []
        for u in split_hidden_states:
            seq_len = u.shape[0]
            if seq_len < max_seq_len:
                # mx.pad() is more efficient than zeros + concatenate
                pad_width = [(0, max_seq_len - seq_len), (0, 0)]
                padded = mx.pad(u, pad_width, constant_values=0.0)
            else:
                padded = u
            padded_embeds.append(padded)

        prompt_embeds = mx.stack(padded_embeds, axis=0)

        # Same optimization for attention masks
        padded_masks = []
        for mask in attn_mask_list:
            seq_len = mask.shape[0]
            if seq_len < max_seq_len:
                # mx.pad() for 1D array
                pad_width = [(0, max_seq_len - seq_len)]
                padded = mx.pad(mask, pad_width, constant_values=0)
            else:
                padded = mask
            padded_masks.append(padded)

        encoder_attention_mask = mx.stack(padded_masks, axis=0)
        prompt_embeds = prompt_embeds.astype(dtype)
        return prompt_embeds, encoder_attention_mask

    @staticmethod
    def _extract_masked_hidden(hidden_states, attention_mask):
        """
        Extract masked hidden states with minimal GPU-CPU synchronization.

        OPTIMIZATION: Compute valid lengths on GPU in single operation,
        then extract with minimal .item() calls (one per batch instead of one per layer).
        This eliminates 28-layer × batch_size synchronization points.
        """
        batch_size = hidden_states.shape[0]

        # Compute all valid lengths on GPU in single operation (no sync yet)
        valid_lengths = mx.sum(attention_mask, axis=1)  # [batch_size]

        # Extract variable-length sequences
        # NOTE: We still need .item() for variable-length slicing, but only once per batch element
        # instead of once per batch per layer (28 layers × batch = massive reduction)
        split_hidden_states = []
        max_seq_len = hidden_states.shape[1]
        for i in range(batch_size):
            # Single .item() call per batch element (not per layer as before)
            length_idx = int(valid_lengths[i].item())
            # CRITICAL FIX: Validate bounds to prevent corruption from invalid attention masks
            length_idx = max(0, min(length_idx, max_seq_len))
            valid_hidden = hidden_states[i, :length_idx, :]
            split_hidden_states.append(valid_hidden)

        return split_hidden_states
