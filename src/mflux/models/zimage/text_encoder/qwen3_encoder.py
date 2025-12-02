import mlx.core as mx
import mlx.nn as nn

from mflux.models.zimage.text_encoder.qwen3_layer import Qwen3DecoderLayer


class Qwen3Encoder(nn.Module):
    """Qwen3-4B text encoder for Z-Image.

    Full 36-layer decoder-only transformer used as text encoder.
    Uses last hidden states (not logits) as text embeddings.
    """

    # Qwen3-4B architecture
    HIDDEN_SIZE = 2560
    NUM_LAYERS = 36
    VOCAB_SIZE = 151936
    RMS_NORM_EPS = 1e-6
    MAX_SEQ_LEN = 2048

    def __init__(self):
        super().__init__()

        # Token embeddings
        self.embed_tokens = nn.Embedding(self.VOCAB_SIZE, self.HIDDEN_SIZE)

        # Decoder layers
        self.layers = [Qwen3DecoderLayer() for _ in range(self.NUM_LAYERS)]

        # Final layer norm
        self.norm = nn.RMSNorm(self.HIDDEN_SIZE, eps=self.RMS_NORM_EPS)

    def __call__(
        self,
        input_ids: mx.array,
        attention_mask: mx.array | None = None,
        return_layer: int = -2,
    ) -> mx.array:
        """Forward pass to get text embeddings.

        Args:
            input_ids: Token IDs [B, seq_len]
            attention_mask: Optional attention mask [B, seq_len], 1=attend, 0=mask
            return_layer: Which layer's output to return (default -2 matches diffusers)
                          -1 = final normed output, -2 = output before final layer

        Returns:
            Hidden states [B, seq_len, 2560]
        """
        # Embed tokens
        hidden_states = self.embed_tokens(input_ids)

        # Create causal mask
        seq_len = input_ids.shape[1]
        causal_mask = mx.triu(mx.full((seq_len, seq_len), -1e9), k=1)

        # Apply attention mask if provided
        if attention_mask is not None:
            # Convert 1D mask to 2D attention mask
            # attention_mask: [B, seq_len], 1=attend, 0=mask
            # Need: [B, 1, 1, seq_len] for broadcasting with attention scores
            mask_2d = (1 - attention_mask[:, None, None, :]) * -1e9
            # Broadcast causal mask to batch dimension and add padding mask
            causal_mask = causal_mask[None, None, :, :] + mask_2d
        else:
            # Just broadcast causal mask for batch compatibility
            causal_mask = causal_mask[None, None, :, :]

        # Process through layers, collecting hidden states
        # hidden_states list: [embeddings, layer_0_out, layer_1_out, ..., layer_35_out]
        all_hidden_states = [hidden_states]
        for layer in self.layers:
            hidden_states = layer(hidden_states, mask=causal_mask)
            all_hidden_states.append(hidden_states)

        # Return the requested layer's output
        if return_layer == -1:
            # Final normed output (original behavior)
            return self.norm(hidden_states)
        else:
            # Diffusers uses hidden_states[-2] which is output of layer 34 (second-to-last)
            # Our list: [embed, l0, l1, ..., l34, l35]  (len=37)
            # -2 index = l34 output
            return all_hidden_states[return_layer]

    def encode(
        self,
        input_ids: mx.array,
        attention_mask: mx.array | None = None,
    ) -> mx.array:
        """Convenience method that returns text embeddings.

        Same as __call__ but with clearer semantics.
        """
        return self(input_ids, attention_mask)
