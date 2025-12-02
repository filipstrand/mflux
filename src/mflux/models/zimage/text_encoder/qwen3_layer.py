import mlx.core as mx
import mlx.nn as nn

from mflux.models.zimage.text_encoder.qwen3_attention import Qwen3Attention
from mflux.models.zimage.text_encoder.qwen3_mlp import Qwen3MLP


class Qwen3DecoderLayer(nn.Module):
    """Single Qwen3 decoder layer.

    Pre-norm architecture with RMSNorm.
    """

    HIDDEN_SIZE = 2560
    RMS_NORM_EPS = 1e-6

    def __init__(self):
        super().__init__()

        self.input_layernorm = nn.RMSNorm(self.HIDDEN_SIZE, eps=self.RMS_NORM_EPS)
        self.self_attn = Qwen3Attention()
        self.post_attention_layernorm = nn.RMSNorm(self.HIDDEN_SIZE, eps=self.RMS_NORM_EPS)
        self.mlp = Qwen3MLP()

    def __call__(self, x: mx.array, mask: mx.array | None = None) -> mx.array:
        """Forward pass with pre-norm and residual connections.

        Args:
            x: Input tensor [B, S, HIDDEN_SIZE]
            mask: Optional attention mask

        Returns:
            Output tensor [B, S, HIDDEN_SIZE]
        """
        # Pre-norm attention
        residual = x
        x = self.input_layernorm(x)
        x = self.self_attn(x, mask=mask)
        x = residual + x

        # Pre-norm MLP
        residual = x
        x = self.post_attention_layernorm(x)
        x = self.mlp(x)
        x = residual + x

        return x
