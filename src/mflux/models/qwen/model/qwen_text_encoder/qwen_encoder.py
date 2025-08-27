import mlx.core as mx
from mlx import nn

from mflux.models.qwen.model.qwen_text_encoder.qwen_encoder_layer import QwenEncoderLayer
from mflux.models.qwen.model.qwen_text_encoder.qwen_rms_norm import QwenRMSNorm


class QwenEncoder(nn.Module):
    def __init__(
        self,
        vocab_size: int = 152064,
        hidden_size: int = 3584,
        num_hidden_layers: int = 28,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.embed_tokens = nn.Embedding(vocab_size, hidden_size)
        self.layers = [QwenEncoderLayer() for i in range(num_hidden_layers)]
        self.norm = QwenRMSNorm(hidden_size, eps=1e-6)

    def __call__(
        self,
        input_ids: mx.array,
        attention_mask: mx.array | None = None,
        position_ids: mx.array | None = None,
    ) -> mx.array:
        hidden_states = self.embed_tokens(input_ids)
        for layer in self.layers:
            hidden_states = layer(hidden_states, attention_mask, position_ids)
        hidden_states = self.norm(hidden_states)
        return hidden_states
