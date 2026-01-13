"""Mistral3 Text Encoder for FLUX.2.

Mistral3 is a multimodal encoder with:
- 40 transformer layers
- GQA with 32 query heads, 8 KV heads
- 5120 hidden size
- 131072 vocabulary size

For FLUX.2, the output is projected to joint_attention_dim (15360).
"""

import mlx.core as mx
from mlx import nn

from mflux.models.flux2.model.flux2_text_encoder.mistral3_encoder_layer import Mistral3EncoderLayer
from mflux.models.flux2.model.flux2_text_encoder.mistral3_rms_norm import Mistral3RMSNorm
from mflux.models.flux2.model.flux2_text_encoder.mistral3_rope import Mistral3RotaryEmbedding


class Mistral3TextEncoder(nn.Module):
    """Mistral3 Text Encoder for FLUX.2.

    Args:
        vocab_size: Vocabulary size (131072)
        hidden_size: Hidden dimension (5120)
        num_hidden_layers: Number of transformer layers (40)
        num_attention_heads: Number of query heads (32)
        num_key_value_heads: Number of KV heads for GQA (8)
        intermediate_size: MLP intermediate dimension (32768)
        max_position_embeddings: Maximum sequence length (131072)
        rms_norm_eps: RMSNorm epsilon (1e-5)
        rope_theta: RoPE base frequency (1_000_000_000)
        joint_attention_dim: Output projection dimension for FLUX.2 (15360)
    """

    def __init__(
        self,
        vocab_size: int = 131072,
        hidden_size: int = 5120,
        num_hidden_layers: int = 40,
        num_attention_heads: int = 32,
        num_key_value_heads: int = 8,
        intermediate_size: int = 32768,
        max_position_embeddings: int = 131072,
        rms_norm_eps: float = 1e-5,
        rope_theta: float = 1_000_000_000.0,
        joint_attention_dim: int = 15360,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.joint_attention_dim = joint_attention_dim

        # Token embeddings
        self.embed_tokens = nn.Embedding(vocab_size, hidden_size)

        # Transformer layers
        self.layers = [
            Mistral3EncoderLayer(
                hidden_size=hidden_size,
                num_heads=num_attention_heads,
                num_kv_heads=num_key_value_heads,
                intermediate_size=intermediate_size,
                rms_norm_eps=rms_norm_eps,
            )
            for _ in range(num_hidden_layers)
        ]

        # Final layer norm
        self.norm = Mistral3RMSNorm(hidden_size, eps=rms_norm_eps)

        # RoPE embeddings
        self.rotary_emb = Mistral3RotaryEmbedding(
            dim=128,  # head_dim
            max_position_embeddings=max_position_embeddings,
            base=rope_theta,
        )

        # Output projection to joint_attention_dim for FLUX.2
        self.output_proj = nn.Linear(hidden_size, joint_attention_dim, bias=False)

    def __call__(
        self,
        input_ids: mx.array,
        attention_mask: mx.array,
    ) -> tuple[mx.array, mx.array]:
        """Encode text and return sequence and pooled embeddings.

        Args:
            input_ids: Token IDs [B, seq_len]
            attention_mask: Attention mask [B, seq_len]

        Returns:
            Tuple of:
            - prompt_embeds: Sequence embeddings [B, seq_len, joint_attention_dim]
            - pooled_prompt_embeds: Pooled embeddings [B, hidden_size]
        """
        batch_size, seq_len = input_ids.shape

        # Get token embeddings
        hidden_states = self.embed_tokens(input_ids)

        # Create position IDs
        position_ids = mx.arange(seq_len, dtype=mx.int32)
        position_ids = mx.expand_dims(position_ids, axis=0)
        position_ids = mx.broadcast_to(position_ids, (batch_size, seq_len))

        # Compute RoPE embeddings
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        # Create causal attention mask
        attention_mask_4d = self._create_causal_mask(attention_mask, seq_len)

        # Run through transformer layers
        for layer in self.layers:
            hidden_states = layer(
                hidden_states=hidden_states,
                attention_mask=attention_mask_4d,
                position_embeddings=position_embeddings,
            )

        # Apply final layer norm
        hidden_states = self.norm(hidden_states)

        # Compute pooled embeddings (mean pooling over valid tokens)
        pooled_prompt_embeds = self._compute_pooled_embeddings(hidden_states, attention_mask)

        # Project to joint_attention_dim
        prompt_embeds = self.output_proj(hidden_states)

        return prompt_embeds.astype(mx.bfloat16), pooled_prompt_embeds.astype(mx.bfloat16)

    def _create_causal_mask(self, attention_mask: mx.array, seq_len: int) -> mx.array:
        """Create 4D causal attention mask.

        Args:
            attention_mask: Padding mask [B, seq_len]
            seq_len: Sequence length

        Returns:
            4D attention mask [B, 1, seq_len, seq_len]
        """
        batch_size = attention_mask.shape[0]

        # Create causal mask
        idx = mx.arange(seq_len, dtype=mx.int32)
        j = mx.expand_dims(idx, axis=0)
        i = mx.expand_dims(idx, axis=1)
        causal_mask = mx.where(j > i, -float("inf"), 0.0).astype(mx.float32)
        causal_mask = mx.expand_dims(mx.expand_dims(causal_mask, axis=0), axis=0)
        causal_mask = mx.broadcast_to(causal_mask, (batch_size, 1, seq_len, seq_len))

        # Create padding mask
        padding_mask = mx.where(
            attention_mask == 1,
            mx.zeros_like(attention_mask).astype(mx.float32),
            mx.ones_like(attention_mask).astype(mx.float32) * (-float("inf")),
        )
        padding_mask = mx.expand_dims(mx.expand_dims(padding_mask, axis=1), axis=1)

        # Combine masks
        attention_mask_4d = causal_mask + padding_mask

        return attention_mask_4d

    def _compute_pooled_embeddings(
        self,
        hidden_states: mx.array,
        attention_mask: mx.array,
    ) -> mx.array:
        """Compute pooled embeddings via mean pooling.

        Args:
            hidden_states: Hidden states [B, seq_len, hidden_size]
            attention_mask: Attention mask [B, seq_len]

        Returns:
            Pooled embeddings [B, hidden_size]
        """
        # Expand mask for broadcasting
        mask_expanded = mx.expand_dims(attention_mask, axis=-1).astype(hidden_states.dtype)

        # Compute weighted sum
        sum_embeddings = mx.sum(hidden_states * mask_expanded, axis=1)
        sum_mask = mx.sum(mask_expanded, axis=1)

        # Avoid division by zero
        sum_mask = mx.maximum(sum_mask, 1e-9)

        pooled = sum_embeddings / sum_mask
        return pooled
