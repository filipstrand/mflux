import mlx.core as mx
from mlx import nn

from mflux.models.qwen.model.qwen_vae.qwen_image_rms_norm import QwenImageRMSNorm


class QwenImageAttentionBlock3D(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        self.norm = QwenImageRMSNorm(dim)
        self.to_qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, stride=1, padding=0)
        self.proj = nn.Conv2d(dim, dim, kernel_size=1, stride=1, padding=0)

    def __call__(self, x: mx.array) -> mx.array:
        identity = x
        batch_size, channels, time, height, width = x.shape

        # Apply normalization (works on 5D tensors)
        x = self.norm(x)

        # Reshape to apply 2D convolutions
        x_2d = mx.reshape(x, (batch_size * time, channels, height, width))

        # Convert to MLX format for Conv2d
        x_2d = mx.transpose(x_2d, (0, 2, 3, 1))

        # Compute QKV using 2D conv (1x1 kernel acts like linear transformation)
        # Transpose weights from PyTorch format (out_ch, in_ch, h, w) to MLX format (out_ch, h, w, in_ch)
        original_weight = self.to_qkv.weight
        if len(original_weight.shape) == 4:
            mlx_weight = mx.transpose(original_weight, (0, 2, 3, 1))
            self.to_qkv.weight = mlx_weight
            qkv_2d = self.to_qkv(x_2d)
            self.to_qkv.weight = original_weight
        else:
            qkv_2d = self.to_qkv(x_2d)

        # Convert back to channels-first
        qkv_2d = mx.transpose(qkv_2d, (0, 3, 1, 2))

        # Reshape for attention computation
        qkv_2d = mx.reshape(qkv_2d, (batch_size * time, channels * 3, height * width))
        qkv_2d = mx.transpose(qkv_2d, (0, 2, 1))
        qkv_2d = mx.expand_dims(qkv_2d, axis=1)

        # Split into Q, K, V
        q, k, v = mx.split(qkv_2d, 3, axis=-1)

        # Apply scaled dot-product attention
        scale = 1.0 / mx.sqrt(mx.array(float(channels)))
        scores = mx.matmul(q, mx.transpose(k, (0, 1, 3, 2))) * scale
        attn_weights = mx.softmax(scores, axis=-1)
        attn_out = mx.matmul(attn_weights, v)

        # Reshape back to 2D conv format
        attn_out = mx.squeeze(attn_out, axis=1)
        attn_out = mx.transpose(attn_out, (0, 2, 1))
        attn_out = mx.reshape(attn_out, (batch_size * time, channels, height, width))

        # Convert to MLX format for Conv2d
        attn_out = mx.transpose(attn_out, (0, 2, 3, 1))

        # Output projection using 2D conv
        # Transpose weights from PyTorch format (out_ch, in_ch, h, w) to MLX format (out_ch, h, w, in_ch)
        original_weight = self.proj.weight
        if len(original_weight.shape) == 4:
            mlx_weight = mx.transpose(original_weight, (0, 2, 3, 1))
            self.proj.weight = mlx_weight
            attn_out = self.proj(attn_out)
            self.proj.weight = original_weight
        else:
            attn_out = self.proj(attn_out)

        # Convert back to channels-first
        attn_out = mx.transpose(attn_out, (0, 3, 1, 2))

        # Reshape back to 5D format
        attn_out = mx.reshape(attn_out, (batch_size, time, channels, height, width))
        attn_out = mx.transpose(attn_out, (0, 2, 1, 3, 4))
        return attn_out + identity
