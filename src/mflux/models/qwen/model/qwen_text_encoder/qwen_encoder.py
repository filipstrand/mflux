import mlx.core as mx
from mlx import nn

from mflux.models.qwen.model.qwen_text_encoder.qwen_encoder_layer import QwenEncoderLayer
from mflux.models.qwen.model.qwen_text_encoder.qwen_rms_norm import QwenRMSNorm
from mflux.models.qwen.model.qwen_text_encoder.qwen_rope import QwenRotaryEmbedding


class QwenEncoder(nn.Module):
    def __init__(
        self,
        vocab_size: int = 152064,
        hidden_size: int = 3584,
        num_hidden_layers: int = 28,
        max_position_embeddings: int = 128000,
        rope_theta: float = 1000000.0,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.embed_tokens = nn.Embedding(vocab_size, hidden_size)
        self.layers = [QwenEncoderLayer() for i in range(num_hidden_layers)]
        self.norm = QwenRMSNorm(hidden_size, eps=1e-6)
        self.rotary_emb = QwenRotaryEmbedding(
            dim=hidden_size // 28,
            max_position_embeddings=max_position_embeddings,
            base=rope_theta,
            rope_type="default",
        )

    def __call__(
        self,
        input_ids: mx.array,
        attention_mask: mx.array | None = None,
    ) -> mx.array:
        batch_size, seq_len = input_ids.shape
        inputs_embeds = self.embed_tokens(input_ids)
        cache_position = mx.arange(seq_len, dtype=mx.int32)
        position_ids = mx.expand_dims(mx.expand_dims(cache_position, axis=0), axis=0)  # [1, 1, seq_len]
        position_ids = mx.broadcast_to(position_ids, (3, batch_size, seq_len))  # [3, batch_size, seq_len]
        if attention_mask is not None:
            padding_mask = mx.where(
                attention_mask == 1,
                mx.zeros_like(attention_mask).astype(mx.float32),
                mx.ones_like(attention_mask).astype(mx.float32) * (-float("inf")),
            )
            padding_mask = mx.expand_dims(mx.expand_dims(padding_mask, axis=1), axis=1)
        else:
            padding_mask = None

        # Create causal triangular mask [batch, 1, seq_len, seq_len]
        idx = mx.arange(seq_len, dtype=mx.int32)
        j = mx.expand_dims(idx, axis=0)  # (1, S)
        i = mx.expand_dims(idx, axis=1)  # (S, 1)
        tri_bool = j > i
        zeros_2d = mx.zeros((seq_len, seq_len)).astype(mx.float32)
        neginf_2d = mx.ones((seq_len, seq_len)).astype(mx.float32) * (-float("inf"))
        causal_tri_mask = mx.where(tri_bool, neginf_2d, zeros_2d)
        causal_tri_mask = mx.expand_dims(mx.expand_dims(causal_tri_mask, axis=0), axis=0)  # [1,1,S,S]
        causal_tri_mask = mx.broadcast_to(causal_tri_mask, (batch_size, 1, seq_len, seq_len))

        # Final attention mask: causal triangle + padding mask
        if padding_mask is not None:
            attention_mask_4d = causal_tri_mask + padding_mask
        else:
            attention_mask_4d = causal_tri_mask

        hidden_states = inputs_embeds
        position_embeddings = self.rotary_emb(hidden_states, position_ids)
        for layer in self.layers:
            hidden_states = layer(hidden_states, attention_mask_4d, position_embeddings)
        hidden_states = self.norm(hidden_states)
        return hidden_states
