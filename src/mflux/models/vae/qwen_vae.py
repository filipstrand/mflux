"""
Qwen Image VAE Implementation for mflux

This implementation is based on execution trace analysis of the diffusers reference
implementation to focus only on the code paths that are actually used.

Architecture (from trace analysis):
- Channel progression: 16 -> 384 -> 192 -> 96 -> 3
- Spatial progression: 16x16 -> 32x32 -> 64x64 -> 128x128 (8x upsampling)
- Uses 3D causal convolutions, RMS normalization, and residual blocks
- Skips unused features: cache_x, dropout, complex caching
"""

import mlx.core as mx
from mlx import nn


class QwenImageCausalConv3D(nn.Module):
    """
    Simplified 3D causal convolution based on trace analysis.

    From trace: cache_x is always None, so we can skip caching logic.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
    ):
        super().__init__()

        # Standard 3D convolution (MLX format)
        self.conv3d = nn.Conv3d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=0,  # We'll handle padding manually
        )

        # Causal padding setup
        self.padding = padding

    def __call__(self, x: mx.array) -> mx.array:
        """
        Forward pass - matching reference implementation exactly.

        Args:
            x: Input tensor [B, C, T, H, W]
        Returns:
            Output tensor after 3D convolution
        """
        # Apply causal padding matching the reference implementation
        # Reference uses: _padding = (padding[2], padding[2], padding[1], padding[1], 2 * padding[0], 0)
        # PyTorch F.pad format: (pad_left, pad_right, pad_top, pad_bottom, pad_front, pad_back)
        # MLX pad format: [(dim0_left, dim0_right), (dim1_left, dim1_right), ...]

        # Handle both int and tuple padding
        if isinstance(self.padding, int):
            pad_t = pad_h = pad_w = self.padding
        else:
            pad_t, pad_h, pad_w = self.padding

        if pad_t > 0 or pad_h > 0 or pad_w > 0:
            # Padding specification for each dimension:
            # - W dimension (dim 4): pad left=pad_w, right=pad_w
            # - H dimension (dim 3): pad top=pad_h, bottom=pad_h
            # - T dimension (dim 2): pad front=2*pad_t, back=0 (causal)
            # - C dimension (dim 1): no padding
            # - B dimension (dim 0): no padding
            pad_spec = [
                (0, 0),  # Batch dimension
                (0, 0),  # Channel dimension
                (2 * pad_t, 0),  # Temporal dimension (causal: 2*padding, 0)
                (pad_h, pad_h),  # Height dimension
                (pad_w, pad_w),  # Width dimension
            ]
            x = mx.pad(x, pad_spec)

        # Convert to channels-last for MLX Conv3d: [B, C, T, H, W] -> [B, T, H, W, C]
        x = mx.transpose(x, (0, 2, 3, 4, 1))

        # Apply 3D convolution
        x = self.conv3d(x)

        # Convert back to channels-first: [B, T, H, W, C] -> [B, C, T, H, W]
        x = mx.transpose(x, (0, 4, 1, 2, 3))

        return x


class QwenImageRMSNorm(nn.Module):
    """
    RMS normalization based on trace analysis.
    """

    def __init__(self, num_channels: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        # Match reference: scale = sqrt(dim)
        self.scale = float(num_channels) ** 0.5
        # Learnable scale parameter
        self.weight = mx.ones((num_channels,))

    def __call__(self, x: mx.array) -> mx.array:
        """
        Apply RMS normalization to channels dimension.

        Args:
            x: Input tensor [B, C, T, H, W]
        Returns:
            Normalized tensor
        """
        # Reference uses L2 normalization across channel dimension with scaling:
        # F.normalize(x, dim=1) * sqrt(dim) * gamma
        # Compute L2 norm across channels
        l2 = mx.sqrt(mx.sum(mx.square(x), axis=1, keepdims=True) + self.eps)
        x_normalized = (x / l2) * self.scale

        # Apply learnable gamma (broadcast)
        weight = self.weight.reshape(1, -1, 1, 1, 1)
        return x_normalized * weight


class QwenImageResBlock3D(nn.Module):
    """
    3D Residual block based on trace analysis.
    """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()

        self.norm1 = QwenImageRMSNorm(in_channels)
        self.conv1 = QwenImageCausalConv3D(in_channels, out_channels, 3, 1, 1)

        self.norm2 = QwenImageRMSNorm(out_channels)
        self.conv2 = QwenImageCausalConv3D(out_channels, out_channels, 3, 1, 1)

        # Skip connection
        if in_channels != out_channels:
            self.skip_conv = QwenImageCausalConv3D(in_channels, out_channels, 1, 1, 0)
        else:
            self.skip_conv = None

    def __call__(self, x: mx.array) -> mx.array:
        residual = x

        # First conv block
        x = self.norm1(x)
        x = nn.silu(x)
        x = self.conv1(x)

        # Second conv block
        x = self.norm2(x)
        x = nn.silu(x)
        x = self.conv2(x)

        # Skip connection
        if self.skip_conv is not None:
            residual = self.skip_conv(residual)

        return x + residual


class QwenImageResample3D(nn.Module):
    """
    3D resampling module matching the reference implementation.

    For upsample3d mode:
    - time_conv: 3D conv that expands channels (dim → dim * 2)
    - resample: 2D upsample + conv that reduces channels (dim → dim // 2)
    """

    def __init__(self, dim: int, mode: str):
        super().__init__()
        self.dim = dim
        self.mode = mode

        if mode == "upsample3d":
            # time_conv: 3D causal conv that expands channels
            # Kernel size (3,1,1) - only temporal convolution, no spatial
            self.time_conv = QwenImageCausalConv3D(dim, dim * 2, kernel_size=(3, 1, 1), stride=1, padding=(1, 0, 0))

            # resample: 2D nearest upsample + conv that reduces channels
            # The resample conv always expects dim input channels (not dim*2)
            # The time_conv expansion is handled differently in the forward pass
            self.resample_conv = nn.Conv2d(dim, dim // 2, kernel_size=3, stride=1, padding=1)
        elif mode == "upsample2d":
            # For 2D mode, just the resample part - reduce dim → dim//2
            self.resample_conv = nn.Conv2d(dim, dim // 2, kernel_size=3, stride=1, padding=1)
        else:
            raise ValueError(f"Unsupported resample mode: {mode}")

    def _upsample_nearest_2x(self, x: mx.array) -> mx.array:
        """2x nearest neighbor upsampling for spatial dimensions."""
        # x shape: [B*T, H, W, C] (MLX format)
        B_T, H, W, C = x.shape
        # Repeat each pixel 2x2
        x = mx.repeat(x, 2, axis=1)  # Double height: [B*T, 2*H, W, C]
        x = mx.repeat(x, 2, axis=2)  # Double width: [B*T, 2*H, 2*W, C]
        return x

    def __call__(self, x: mx.array) -> mx.array:
        b, c, t, h, w = x.shape

        if self.mode == "upsample3d":
            # For single-frame inference (no feat_cache), the reference implementation
            # SKIPS the time_conv entirely and goes directly to spatial processing.
            # The time_conv is only used for multi-frame temporal processing with caching.
            # So upsample3d behaves exactly like upsample2d for single frames.
            pass  # Skip time_conv for single-frame inference

        # Reshape to 2D format for spatial processing
        t = x.shape[2]
        x = mx.transpose(x, (0, 2, 1, 3, 4))  # [B, T, C, H, W]
        x = mx.reshape(x, (b * t, c, h, w))  # [B*T, C, H, W]

        # Convert to MLX Conv2d format and apply resample conv
        x = mx.transpose(x, (0, 2, 3, 1))  # [B*T, H, W, C]
        x = self._upsample_nearest_2x(x)  # [B*T, 2*H, 2*W, C]
        x = self.resample_conv(x)  # [B*T, 2*H, 2*W, C//2]
        x = mx.transpose(x, (0, 3, 1, 2))  # [B*T, C//2, 2*H, 2*W]

        # Reshape back to 5D format
        new_c = x.shape[1]
        new_h, new_w = x.shape[2], x.shape[3]
        x = mx.reshape(x, (b, t, new_c, new_h, new_w))  # [B, T, C//2, 2*H, 2*W]
        x = mx.transpose(x, (0, 2, 1, 3, 4))  # [B, C//2, T, 2*H, 2*W]

        return x


class QwenImageUpBlock3D(nn.Module):
    """
    3D upsampling block matching the reference implementation.
    """

    def __init__(self, in_channels: int, out_channels: int, num_res_blocks: int = 2, upsample_mode: str = None):
        super().__init__()

        # Residual blocks (num_res_blocks + 1, as per reference)
        self.resnets = []
        current_dim = in_channels
        for _ in range(num_res_blocks + 1):
            self.resnets.append(QwenImageResBlock3D(current_dim, out_channels))
            current_dim = out_channels

        # Upsampler (if needed)
        self.upsamplers = None
        if upsample_mode is not None:
            self.upsamplers = [QwenImageResample3D(out_channels, mode=upsample_mode)]

    def __call__(self, x: mx.array) -> mx.array:
        # Apply residual blocks
        for resnet in self.resnets:
            x = resnet(x)

        # Apply upsampler if present
        if self.upsamplers is not None:
            x = self.upsamplers[0](x)

        return x


class QwenImageAttentionBlock3D(nn.Module):
    """
    3D attention block for Qwen Image VAE.
    Based on reference QwenImageAttentionBlock but uses 2D convolutions as per reference.
    """

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

        # Components - use 1x1 2D convolutions as per reference implementation
        self.norm = QwenImageRMSNorm(dim)
        self.to_qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, stride=1, padding=0)
        self.proj = nn.Conv2d(dim, dim, kernel_size=1, stride=1, padding=0)

    def __call__(self, x: mx.array) -> mx.array:
        """
        Apply attention to 5D tensor [B, C, T, H, W].
        Uses 2D convolutions on spatial dimensions, handling temporal dimension separately.
        """
        identity = x
        batch_size, channels, time, height, width = x.shape

        # Apply normalization (works on 5D tensors)
        x = self.norm(x)

        # Reshape to apply 2D convolutions: [B, C, T, H, W] -> [B*T, C, H, W]
        x_2d = mx.reshape(x, (batch_size * time, channels, height, width))

        # Convert to MLX format for Conv2d: [B*T, C, H, W] -> [B*T, H, W, C]
        x_2d = mx.transpose(x_2d, (0, 2, 3, 1))

        # Compute QKV using 2D conv (1x1 kernel acts like linear transformation)
        qkv_2d = self.to_qkv(x_2d)  # [B*T, H, W, C*3]

        # Convert back to channels-first: [B*T, H, W, C*3] -> [B*T, C*3, H, W]
        qkv_2d = mx.transpose(qkv_2d, (0, 3, 1, 2))

        # Reshape for attention computation
        qkv_2d = mx.reshape(qkv_2d, (batch_size * time, channels * 3, height * width))
        qkv_2d = mx.transpose(qkv_2d, (0, 2, 1))  # [B*T, H*W, C*3]
        qkv_2d = mx.expand_dims(qkv_2d, axis=1)  # [B*T, 1, H*W, C*3]

        # Split into Q, K, V
        q, k, v = mx.split(qkv_2d, 3, axis=-1)  # Each: [B*T, 1, H*W, C]

        # Apply scaled dot-product attention
        scale = 1.0 / mx.sqrt(mx.array(float(channels)))
        scores = mx.matmul(q, mx.transpose(k, (0, 1, 3, 2))) * scale
        attn_weights = mx.softmax(scores, axis=-1)
        attn_out = mx.matmul(attn_weights, v)  # [B*T, 1, H*W, C]

        # Reshape back to 2D conv format
        attn_out = mx.squeeze(attn_out, axis=1)  # [B*T, H*W, C]
        attn_out = mx.transpose(attn_out, (0, 2, 1))  # [B*T, C, H*W]
        attn_out = mx.reshape(attn_out, (batch_size * time, channels, height, width))  # [B*T, C, H, W]

        # Convert to MLX format for Conv2d: [B*T, C, H, W] -> [B*T, H, W, C]
        attn_out = mx.transpose(attn_out, (0, 2, 3, 1))

        # Output projection using 2D conv
        attn_out = self.proj(attn_out)  # [B*T, H, W, C]

        # Convert back to channels-first: [B*T, H, W, C] -> [B*T, C, H, W]
        attn_out = mx.transpose(attn_out, (0, 3, 1, 2))

        # Reshape back to 5D format
        attn_out = mx.reshape(attn_out, (batch_size, time, channels, height, width))  # [B, T, C, H, W]
        attn_out = mx.transpose(attn_out, (0, 2, 1, 3, 4))  # [B, C, T, H, W]

        return attn_out + identity


class QwenImageMidBlock3D(nn.Module):
    """
    Middle block for Qwen Image VAE decoder.
    Based on reference QwenImageMidBlock with proper ResNet + Attention structure.
    """

    def __init__(self, dim: int, num_layers: int = 1):
        super().__init__()
        self.dim = dim

        # Create components like reference:
        # - 1 initial ResNet
        # - For each attention layer: 1 attention + 1 ResNet
        resnets = [QwenImageResBlock3D(dim, dim)]
        attentions = []

        for _ in range(num_layers):
            attentions.append(QwenImageAttentionBlock3D(dim))
            resnets.append(QwenImageResBlock3D(dim, dim))

        self.attentions = attentions
        self.resnets = resnets

    def __call__(self, x: mx.array) -> mx.array:
        """
        Forward pass matching reference implementation:
        ResNet -> (Attention -> ResNet) * num_layers
        """
        # First residual block
        x = self.resnets[0](x)

        # Process through attention and residual blocks
        for attn, resnet in zip(self.attentions, self.resnets[1:]):
            if attn is not None:
                x = attn(x)
            x = resnet(x)

        return x


class QwenImageDecoder3D(nn.Module):
    """
    3D decoder implementation based on execution trace.

    Architecture from trace:
    16 -> 384 -> 192 -> 96 -> 3 channels
    16x16 -> 32x32 -> 64x64 -> 128x128 spatial
    """

    def __init__(self):
        super().__init__()

        # Initial convolution: 16 -> 384 channels
        self.conv_in = QwenImageCausalConv3D(16, 384, 3, 1, 1)

        # Middle block (proper ResNet + Attention structure)
        self.mid_block = QwenImageMidBlock3D(384, num_layers=1)

        # Upsampling blocks based on actual config analysis
        # Config: base_dim=96, dim_mult=[1,2,4,4], temporal_upsample=[False,True,True]
        # Decoder dims: [384, 384, 384, 192, 96] with upsampler channel reduction

        # up_blocks.0: 384 -> 384, upsample3d (temporal_upsample[0] = True)
        self.up_block0 = QwenImageUpBlock3D(384, 384, num_res_blocks=2, upsample_mode="upsample3d")

        # up_blocks.1: 192 -> 384, upsample3d (384→192 channel reduction)
        # Note: First ResNet does 192→384 channel expansion, then upsampler reduces 384→192
        self.up_block1 = QwenImageUpBlock3D(192, 384, num_res_blocks=2, upsample_mode="upsample3d")

        # up_blocks.2: 192 -> 192, upsample2d (192→96 channel reduction)
        # Reference shows up_block2 has no time_conv, so use upsample2d mode
        self.up_block2 = QwenImageUpBlock3D(192, 192, num_res_blocks=2, upsample_mode="upsample2d")

        # up_blocks.3: 96 -> 96, no upsampling (final processing)
        self.up_block3 = QwenImageUpBlock3D(96, 96, num_res_blocks=2, upsample_mode=None)

        # Output layers
        self.norm_out = QwenImageRMSNorm(96)
        self.conv_out = QwenImageCausalConv3D(96, 3, 3, 1, 1)  # RGB output

    def __call__(self, x: mx.array) -> mx.array:
        """
        Decode latents following the traced execution path.

        Args:
            x: Input latents [1, 16, 1, 16, 16]
        Returns:
            Decoded images [1, 3, 1, 128, 128]
        """
        print(f"🔍 Decoder input: {x.shape}")

        # Initial convolution: 16 -> 384 channels
        x = self.conv_in(x)
        print(f"🔍 After conv_in: {x.shape}")

        # Middle block processing
        x = self.mid_block(x)
        print(f"🔍 After mid_block: {x.shape}")

        # up_blocks.0: Process 384 channels and upsample
        x = self.up_block0(x)
        print(f"🔍 After up_block0: {x.shape}")

        # up_blocks.1: Process 192 channels and upsample
        x = self.up_block1(x)
        print(f"🔍 After up_block1: {x.shape}")

        # up_blocks.2: Process 192 channels and upsample to 96
        x = self.up_block2(x)
        print(f"🔍 After up_block2: {x.shape}")

        # up_blocks.3: Final processing at 96 channels, no upsampling
        x = self.up_block3(x)
        print(f"🔍 After up_block3: {x.shape}")

        # Output layers
        x = self.norm_out(x)
        x = nn.silu(x)
        x = self.conv_out(x)
        print(f"🔍 Final output: {x.shape}")

        return x


class QwenVAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.post_quant_conv = QwenImageCausalConv3D(16, 16, 1, 1, 0)
        self.decoder = QwenImageDecoder3D()

    def decode(self, latents: mx.array) -> mx.array:
        latents = self.post_quant_conv(latents)
        return self.decoder(latents)

    def decode_latents(self, latents: mx.array) -> mx.array:
        latents = latents.reshape(latents.shape[0], latents.shape[1], 1, latents.shape[2], latents.shape[3])
        latents_mean = mx.array(
            [
                -0.7571,
                -0.7089,
                -0.9113,
                0.1075,
                -0.1745,
                0.9653,
                -0.1517,
                1.5508,
                0.4134,
                -0.0715,
                0.5517,
                -0.3632,
                -0.1922,
                -0.9497,
                0.2503,
                -0.2921,
            ]
        ).reshape(1, 16, 1, 1, 1)

        latents_std = mx.array(
            [
                2.8184,
                1.4541,
                2.3275,
                2.6558,
                1.2196,
                1.7708,
                2.6052,
                2.0743,
                3.2687,
                2.1526,
                2.8652,
                1.5579,
                1.6382,
                1.1253,
                2.8251,
                1.916,
            ]
        ).reshape(1, 16, 1, 1, 1)
        latents = latents / (1.0 / latents_std) + latents_mean
        decoded = self.decode(latents)
        return decoded[:, :, 0, :, :]
