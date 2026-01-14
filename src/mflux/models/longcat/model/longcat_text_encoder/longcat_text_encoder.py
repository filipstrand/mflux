"""
Text encoder for LongCat-Image model.

Uses Qwen2.5-VL as the text encoder backbone. This module wraps the
QwenEncoder to provide both sequence and pooled embeddings as required
by the LongCat transformer.
"""

import mlx.core as mx
from mlx import nn

from mflux.models.qwen.model.qwen_text_encoder.qwen_encoder import QwenEncoder


class LongCatTextEncoder(nn.Module):
    """
    Text encoder for LongCat-Image model.

    Uses Qwen2.5-VL (3584 hidden, 28 layers) to encode text prompts.
    Provides both sequence embeddings and pooled embeddings.

    Architecture:
    - vocab_size: 152064
    - hidden_size: 3584
    - num_hidden_layers: 28
    - GQA: 28 query heads, 4 KV heads
    """

    def __init__(
        self,
        vocab_size: int = 152064,
        hidden_size: int = 3584,
        num_hidden_layers: int = 28,
        max_position_embeddings: int = 128000,
        rope_theta: float = 1000000.0,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.encoder = QwenEncoder(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            num_hidden_layers=num_hidden_layers,
            max_position_embeddings=max_position_embeddings,
            rope_theta=rope_theta,
        )

    def __call__(
        self,
        input_ids: mx.array,
        attention_mask: mx.array,
    ) -> tuple[mx.array, mx.array]:
        """
        Encode text input and return sequence and pooled embeddings.

        Args:
            input_ids: Token IDs [B, seq_len]
            attention_mask: Attention mask [B, seq_len]

        Returns:
            Tuple of:
            - prompt_embeds: Sequence embeddings [B, seq_len, 3584]
            - pooled_prompt_embeds: Pooled embeddings [B, 3584]
        """
        # Get sequence embeddings from Qwen encoder
        hidden_states = self.encoder(input_ids, attention_mask)

        # Process embeddings
        prompt_embeds, pooled_prompt_embeds = self._process_embeddings(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
        )

        return prompt_embeds, pooled_prompt_embeds

    def _process_embeddings(
        self,
        hidden_states: mx.array,
        attention_mask: mx.array,
    ) -> tuple[mx.array, mx.array]:
        """
        Process encoder hidden states into sequence and pooled embeddings.

        Uses mean pooling over valid tokens for the pooled representation,
        which is a common approach for encoder-only models.

        Args:
            hidden_states: Encoder output [B, seq_len, hidden_size]
            attention_mask: Attention mask [B, seq_len]

        Returns:
            Tuple of (sequence_embeds, pooled_embeds)
        """
        # Sequence embeddings are just the hidden states
        prompt_embeds = hidden_states.astype(mx.bfloat16)

        # Pooled embeddings via mean pooling over valid tokens
        # Expand attention mask for broadcasting
        mask_expanded = mx.expand_dims(attention_mask, axis=-1).astype(hidden_states.dtype)

        # Compute sum and count for mean
        sum_embeddings = mx.sum(hidden_states * mask_expanded, axis=1)
        sum_mask = mx.sum(mask_expanded, axis=1)

        # Avoid division by zero
        sum_mask = mx.maximum(sum_mask, 1e-9)

        pooled_prompt_embeds = (sum_embeddings / sum_mask).astype(mx.bfloat16)

        return prompt_embeds, pooled_prompt_embeds
