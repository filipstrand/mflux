import mlx.core as mx
from mlx import nn

from mflux.models.common.config import ModelConfig
from mflux.models.z_image.model.z_image_text_encoder.encoder_layer import EncoderLayer
from mflux.models.z_image.model.z_image_text_encoder.rope import RotaryEmbedding


class TextEncoder(nn.Module):
    def __init__(
        self,
        vocab_size: int = 151936,
        hidden_size: int = 2560,
        num_hidden_layers: int = 36,
        num_attention_heads: int = 32,
        num_key_value_heads: int = 8,
        intermediate_size: int = 9728,
        head_dim: int = 128,
        max_position_embeddings: int = 40960,
        rope_theta: float = 1000000.0,
        rms_norm_eps: float = 1e-6,
    ):
        super().__init__()
        self.embed_tokens = nn.Embedding(vocab_size, hidden_size)
        self.layers = [
            EncoderLayer(
                hidden_size=hidden_size,
                num_attention_heads=num_attention_heads,
                num_key_value_heads=num_key_value_heads,
                intermediate_size=intermediate_size,
                head_dim=head_dim,
                rms_norm_eps=rms_norm_eps,
            )
            for _ in range(num_hidden_layers)
        ]
        self.norm = nn.RMSNorm(hidden_size, eps=rms_norm_eps)
        self.rotary_emb = RotaryEmbedding(dim=head_dim, base=rope_theta)

    def __call__(self, input_ids: mx.array, attention_mask: mx.array | None = None) -> mx.array:
        batch_size, seq_len = input_ids.shape
        hidden_states = self.embed_tokens(input_ids).astype(mx.float32)
        position_ids = mx.broadcast_to(mx.arange(seq_len, dtype=mx.int32)[None, :], (batch_size, seq_len))
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        # Causal mask
        causal_mask = TextEncoder._get_causal_mask(attention_mask, batch_size, hidden_states, seq_len)

        # Forward through layers
        all_hidden_states = [hidden_states]
        for layer in self.layers:
            hidden_states = layer(
                hidden_states=hidden_states,
                attention_mask=causal_mask,
                position_embeddings=position_embeddings,
            )
            all_hidden_states.append(hidden_states)

        return all_hidden_states[-2].astype(ModelConfig.precision)

    @staticmethod
    def _get_causal_mask(attention_mask, batch_size, hidden_states, seq_len):
        causal_mask = TextEncoder._create_causal_mask(seq_len, hidden_states.dtype)
        if attention_mask is not None:
            padding_mask = mx.where(
                attention_mask[:, None, None, :] == 1,
                mx.zeros((batch_size, 1, 1, seq_len), dtype=hidden_states.dtype),
                mx.full((batch_size, 1, 1, seq_len), float("-inf"), dtype=hidden_states.dtype),
            )
            causal_mask = causal_mask + padding_mask
        return causal_mask

    @staticmethod
    def _create_causal_mask(seq_len: int, dtype: mx.Dtype) -> mx.array:
        idx = mx.arange(seq_len, dtype=mx.int32)
        mask = idx[:, None] >= idx[None, :]
        causal_mask = mx.where(
            mask,
            mx.zeros((seq_len, seq_len), dtype=dtype),
            mx.full((seq_len, seq_len), float("-inf"), dtype=dtype),
        )
        return causal_mask[None, None, :, :]
