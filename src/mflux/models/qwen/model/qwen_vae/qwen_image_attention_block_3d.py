import mlx.core as mx
from mlx import nn

from mflux.models.qwen.model.qwen_vae.qwen_image_rms_norm import QwenImageRMSNorm


class QwenImageAttentionBlock3D(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        self.norm = QwenImageRMSNorm(dim, images=True)
        self.to_qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, stride=1, padding=0)
        self.proj = nn.Conv2d(dim, dim, kernel_size=1, stride=1, padding=0)

    def __call__(self, x: mx.array) -> mx.array:
        identity = x
        batch_size, channels, time, height, width = x.shape
        x = mx.transpose(x, (0, 2, 1, 3, 4))
        x = mx.reshape(x, (batch_size * time, channels, height, width))
        x_5d = mx.expand_dims(x, axis=2)
        x_5d = self.norm(x_5d)
        x = mx.squeeze(x_5d, axis=2)
        x = mx.transpose(x, (0, 2, 3, 1))
        qkv = self.to_qkv(x)
        qkv = mx.transpose(qkv, (0, 3, 1, 2))
        qkv = mx.reshape(qkv, (batch_size * time, 1, channels * 3, height * width))
        qkv = mx.transpose(qkv, (0, 1, 3, 2))
        q, k, v = mx.split(qkv, 3, axis=-1)
        scale = 1.0 / mx.sqrt(mx.array(float(channels)))
        scores = mx.matmul(q, mx.transpose(k, (0, 1, 3, 2))) * scale
        attn_weights = mx.softmax(scores, axis=-1)
        attn_out = mx.matmul(attn_weights, v)
        attn_out = mx.squeeze(attn_out, axis=1)
        attn_out = mx.transpose(attn_out, (0, 2, 1))
        attn_out = mx.reshape(attn_out, (batch_size * time, channels, height, width))
        attn_out = mx.transpose(attn_out, (0, 2, 3, 1))
        attn_out = self.proj(attn_out)
        attn_out = mx.transpose(attn_out, (0, 3, 1, 2))
        attn_out = mx.reshape(attn_out, (batch_size, time, channels, height, width))
        attn_out = mx.transpose(attn_out, (0, 2, 1, 3, 4))
        return attn_out + identity
