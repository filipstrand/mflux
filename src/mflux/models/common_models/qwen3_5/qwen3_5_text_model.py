import mlx.core as mx
from mlx import nn

from mflux.models.common_models.qwen3_5.qwen3_5_decoder_layer import Qwen35DecoderLayer
from mflux.models.common_models.qwen3_5.qwen3_5_rms_norm import Qwen35RMSNorm

# The text tower of Qwen/Qwen3.5-9B (text_encoder/config.json "text_config").
QWEN3_5_9B_TEXT_CONFIG = dict(
    vocab_size=248320,
    hidden_size=4096,
    intermediate_size=12288,
    num_hidden_layers=32,
    full_attention_interval=4,
    num_attention_heads=16,
    num_key_value_heads=4,
    head_dim=256,
    partial_rotary_factor=0.25,
    rope_theta=10000000.0,
    linear_num_key_heads=16,
    linear_num_value_heads=32,
    linear_key_head_dim=128,
    linear_value_head_dim=128,
    linear_conv_kernel_dim=4,
    rms_norm_eps=1e-6,
)


class Qwen35TextModel(nn.Module):
    # Hybrid Qwen3.5 decoder: three Gated DeltaNet layers, then one gated full-attention
    # layer, repeated. Only the forward pass for one unpadded sequence is supported.
    def __init__(self, **overrides):
        super().__init__()
        config = {**QWEN3_5_9B_TEXT_CONFIG, **overrides}
        layer_types = Qwen35TextModel.layer_types_for(config["num_hidden_layers"], config["full_attention_interval"])
        self.embed_tokens = nn.Embedding(config["vocab_size"], config["hidden_size"])
        self.layers = [Qwen35DecoderLayer(layer_type, config) for layer_type in layer_types]
        self.norm = Qwen35RMSNorm(config["hidden_size"], eps=config["rms_norm_eps"])

    def __call__(self, input_ids: mx.array) -> mx.array:
        hidden_states = self.embed_tokens(input_ids)
        for layer in self.layers:
            hidden_states = layer(hidden_states)
        return self.norm(hidden_states)

    @staticmethod
    def layer_types_for(num_hidden_layers: int, full_attention_interval: int) -> list[str]:
        return [
            "full_attention" if (index + 1) % full_attention_interval == 0 else "linear_attention"
            for index in range(num_hidden_layers)
        ]
