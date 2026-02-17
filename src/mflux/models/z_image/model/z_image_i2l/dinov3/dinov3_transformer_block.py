import mlx.core as mx
import mlx.nn as nn

from mflux.models.z_image.model.z_image_i2l.dinov3.dinov3_attention import DINOv3Attention
from mflux.models.z_image.model.z_image_i2l.dinov3.dinov3_layer_scale import DINOv3LayerScale
from mflux.models.z_image.model.z_image_i2l.dinov3.dinov3_mlp import DINOv3GatedMLP


class DINOv3TransformerBlock(nn.Module):
    """Single DINOv3 transformer layer with LayerScale and RoPE."""

    def __init__(self):
        super().__init__()
        self.norm1 = nn.LayerNorm(dims=4096, eps=1e-5)
        self.attention = DINOv3Attention()
        self.layer_scale1 = DINOv3LayerScale(dims=4096, init_values=1.0)
        self.norm2 = nn.LayerNorm(dims=4096, eps=1e-5)
        self.mlp = DINOv3GatedMLP()
        self.layer_scale2 = DINOv3LayerScale(dims=4096, init_values=1.0)

    def __call__(
        self,
        hidden_states: mx.array,
        cos: mx.array,
        sin: mx.array,
        num_prefix_tokens: int = 5,
    ) -> mx.array:
        residual = hidden_states
        hidden_states = self.norm1(hidden_states)
        hidden_states = self.attention(hidden_states, cos, sin, num_prefix_tokens)
        hidden_states = self.layer_scale1(hidden_states)
        hidden_states = hidden_states + residual

        residual = hidden_states
        hidden_states = self.norm2(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = self.layer_scale2(hidden_states)
        hidden_states = hidden_states + residual

        return hidden_states
