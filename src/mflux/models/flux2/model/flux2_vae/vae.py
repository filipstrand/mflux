import mlx.core as mx
from mlx import nn
from mlx.core.fast import scaled_dot_product_attention

from mflux.models.common.config.model_config import ModelConfig


class Flux2BatchNormStats(nn.Module):
    def __init__(self, num_features: int, eps: float = 1e-4, momentum: float = 0.1):
        super().__init__()
        self.running_mean = mx.zeros((num_features,), dtype=mx.float32)
        self.running_var = mx.ones((num_features,), dtype=mx.float32)
        self.eps = eps
        self.momentum = momentum


class Flux2ResnetBlock2D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, eps: float = 1e-6, groups: int = 32):
        super().__init__()
        self.norm1 = nn.GroupNorm(num_groups=groups, dims=in_channels, eps=eps, pytorch_compatible=True)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.norm2 = nn.GroupNorm(num_groups=groups, dims=out_channels, eps=eps, pytorch_compatible=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.conv_shortcut = (
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1) if in_channels != out_channels else None
        )

    def __call__(self, hidden_states: mx.array) -> mx.array:
        residual = mx.transpose(hidden_states, (0, 2, 3, 1))

        hidden_states = mx.transpose(hidden_states, (0, 2, 3, 1))
        hidden_states = self.norm1(hidden_states.astype(mx.float32)).astype(ModelConfig.precision)
        hidden_states = nn.silu(hidden_states)
        hidden_states = self.conv1(hidden_states)
        hidden_states = self.norm2(hidden_states.astype(mx.float32)).astype(ModelConfig.precision)
        hidden_states = nn.silu(hidden_states)
        hidden_states = self.conv2(hidden_states)

        if self.conv_shortcut is not None:
            residual = self.conv_shortcut(residual)

        hidden_states = hidden_states + residual
        return mx.transpose(hidden_states, (0, 3, 1, 2))


class Flux2AttentionBlock(nn.Module):
    def __init__(self, channels: int, groups: int = 32, eps: float = 1e-6):
        super().__init__()
        self.group_norm = nn.GroupNorm(num_groups=groups, dims=channels, eps=eps, pytorch_compatible=True)
        self.to_q = nn.Linear(channels, channels)
        self.to_k = nn.Linear(channels, channels)
        self.to_v = nn.Linear(channels, channels)
        self.to_out = nn.Linear(channels, channels)

    def __call__(self, hidden_states: mx.array) -> mx.array:
        hidden_states = mx.transpose(hidden_states, (0, 2, 3, 1))
        batch, height, width, channels = hidden_states.shape

        normed = self.group_norm(hidden_states.astype(mx.float32)).astype(ModelConfig.precision)
        q = self.to_q(normed).reshape(batch, height * width, 1, channels)
        k = self.to_k(normed).reshape(batch, height * width, 1, channels)
        v = self.to_v(normed).reshape(batch, height * width, 1, channels)

        q = mx.transpose(q, (0, 2, 1, 3))
        k = mx.transpose(k, (0, 2, 1, 3))
        v = mx.transpose(v, (0, 2, 1, 3))

        scale = 1 / mx.sqrt(q.shape[-1])
        attended = scaled_dot_product_attention(q, k, v, scale=scale)
        attended = mx.transpose(attended, (0, 2, 1, 3)).reshape(batch, height, width, channels)
        attended = self.to_out(attended)

        hidden_states = hidden_states + attended
        return mx.transpose(hidden_states, (0, 3, 1, 2))


class Flux2Downsample2D(nn.Module):
    def __init__(self, channels: int, padding: int = 0):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, stride=2, padding=padding)

    def __call__(self, hidden_states: mx.array) -> mx.array:
        hidden_states = mx.transpose(hidden_states, (0, 2, 3, 1))
        hidden_states = self.conv(hidden_states)
        return mx.transpose(hidden_states, (0, 3, 1, 2))


class Flux2Upsample2D(nn.Module):
    def __init__(self, channels: int, out_channels: int | None = None):
        super().__init__()
        out_channels = out_channels or channels
        self.conv = nn.Conv2d(channels, out_channels, kernel_size=3, stride=1, padding=1)

    def __call__(self, hidden_states: mx.array) -> mx.array:
        hidden_states = mx.repeat(hidden_states, 2, axis=2)
        hidden_states = mx.repeat(hidden_states, 2, axis=3)
        hidden_states = mx.transpose(hidden_states, (0, 2, 3, 1))
        hidden_states = self.conv(hidden_states)
        return mx.transpose(hidden_states, (0, 3, 1, 2))


class Flux2DownEncoderBlock2D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_layers: int = 2,
        eps: float = 1e-6,
        groups: int = 32,
        add_downsample: bool = True,
        downsample_padding: int = 0,
    ):
        super().__init__()
        self.resnets = [
            Flux2ResnetBlock2D(
                in_channels=in_channels if i == 0 else out_channels,
                out_channels=out_channels,
                eps=eps,
                groups=groups,
            )
            for i in range(num_layers)
        ]
        self.downsamplers = [Flux2Downsample2D(out_channels, padding=downsample_padding)] if add_downsample else []

    def __call__(self, hidden_states: mx.array) -> mx.array:
        for resnet in self.resnets:
            hidden_states = resnet(hidden_states)
        for downsampler in self.downsamplers:
            hidden_states = downsampler(hidden_states)
        return hidden_states


class Flux2UpDecoderBlock2D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_layers: int = 3,
        eps: float = 1e-6,
        groups: int = 32,
        add_upsample: bool = True,
    ):
        super().__init__()
        self.resnets = [
            Flux2ResnetBlock2D(
                in_channels=in_channels if i == 0 else out_channels,
                out_channels=out_channels,
                eps=eps,
                groups=groups,
            )
            for i in range(num_layers)
        ]
        self.upsamplers = [Flux2Upsample2D(out_channels, out_channels)] if add_upsample else []

    def __call__(self, hidden_states: mx.array) -> mx.array:
        for resnet in self.resnets:
            hidden_states = resnet(hidden_states)
        for upsampler in self.upsamplers:
            hidden_states = upsampler(hidden_states)
        return hidden_states


class Flux2UNetMidBlock2D(nn.Module):
    def __init__(self, channels: int, eps: float = 1e-6, groups: int = 32, add_attention: bool = True):
        super().__init__()
        self.resnets = [
            Flux2ResnetBlock2D(channels, channels, eps=eps, groups=groups),
            Flux2ResnetBlock2D(channels, channels, eps=eps, groups=groups),
        ]
        self.attentions = [Flux2AttentionBlock(channels, groups=groups, eps=eps)] if add_attention else []

    def __call__(self, hidden_states: mx.array) -> mx.array:
        hidden_states = self.resnets[0](hidden_states)
        if self.attentions:
            hidden_states = self.attentions[0](hidden_states)
        hidden_states = self.resnets[1](hidden_states)
        return hidden_states


class Flux2Encoder(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 32,
        block_out_channels: tuple[int, ...] = (128, 256, 512, 512),
        layers_per_block: int = 2,
        norm_num_groups: int = 32,
        eps: float = 1e-6,
        mid_block_add_attention: bool = True,
    ):
        super().__init__()
        self.conv_in = nn.Conv2d(in_channels, block_out_channels[0], kernel_size=3, stride=1, padding=1)
        self.down_blocks = []
        output_channel = block_out_channels[0]
        for i, output_channel in enumerate(block_out_channels):
            input_channel = block_out_channels[i - 1] if i > 0 else block_out_channels[0]
            is_final_block = i == len(block_out_channels) - 1
            self.down_blocks.append(
                Flux2DownEncoderBlock2D(
                    in_channels=input_channel,
                    out_channels=output_channel,
                    num_layers=layers_per_block,
                    eps=eps,
                    groups=norm_num_groups,
                    add_downsample=not is_final_block,
                    downsample_padding=0,
                )
            )

        self.mid_block = Flux2UNetMidBlock2D(
            channels=block_out_channels[-1],
            eps=eps,
            groups=norm_num_groups,
            add_attention=mid_block_add_attention,
        )
        self.conv_norm_out = nn.GroupNorm(
            num_groups=norm_num_groups, dims=block_out_channels[-1], eps=eps, pytorch_compatible=True
        )
        self.conv_out = nn.Conv2d(block_out_channels[-1], 2 * out_channels, kernel_size=3, padding=1)

    def __call__(self, hidden_states: mx.array) -> mx.array:
        hidden_states = mx.transpose(hidden_states, (0, 2, 3, 1))
        hidden_states = self.conv_in(hidden_states)
        hidden_states = mx.transpose(hidden_states, (0, 3, 1, 2))

        for down_block in self.down_blocks:
            hidden_states = down_block(hidden_states)

        hidden_states = self.mid_block(hidden_states)

        hidden_states = mx.transpose(hidden_states, (0, 2, 3, 1))
        hidden_states = self.conv_norm_out(hidden_states.astype(mx.float32)).astype(ModelConfig.precision)
        hidden_states = nn.silu(hidden_states)
        hidden_states = self.conv_out(hidden_states)
        return mx.transpose(hidden_states, (0, 3, 1, 2))


class Flux2Decoder(nn.Module):
    def __init__(
        self,
        in_channels: int = 32,
        out_channels: int = 3,
        block_out_channels: tuple[int, ...] = (128, 256, 512, 512),
        layers_per_block: int = 2,
        norm_num_groups: int = 32,
        eps: float = 1e-6,
        mid_block_add_attention: bool = True,
    ):
        super().__init__()
        self.conv_in = nn.Conv2d(in_channels, block_out_channels[-1], kernel_size=3, stride=1, padding=1)
        self.mid_block = Flux2UNetMidBlock2D(
            channels=block_out_channels[-1],
            eps=eps,
            groups=norm_num_groups,
            add_attention=mid_block_add_attention,
        )
        self.up_blocks = []
        reversed_channels = list(reversed(block_out_channels))
        output_channel = reversed_channels[0]
        for i, output_channel in enumerate(reversed_channels):
            prev_output_channel = output_channel if i == 0 else reversed_channels[i - 1]
            is_final_block = i == len(reversed_channels) - 1
            self.up_blocks.append(
                Flux2UpDecoderBlock2D(
                    in_channels=prev_output_channel,
                    out_channels=output_channel,
                    num_layers=layers_per_block + 1,
                    eps=eps,
                    groups=norm_num_groups,
                    add_upsample=not is_final_block,
                )
            )

        self.conv_norm_out = nn.GroupNorm(
            num_groups=norm_num_groups, dims=block_out_channels[0], eps=eps, pytorch_compatible=True
        )
        self.conv_out = nn.Conv2d(block_out_channels[0], out_channels, kernel_size=3, padding=1)

    def __call__(self, hidden_states: mx.array) -> mx.array:
        hidden_states = mx.transpose(hidden_states, (0, 2, 3, 1))
        hidden_states = self.conv_in(hidden_states)
        hidden_states = mx.transpose(hidden_states, (0, 3, 1, 2))

        hidden_states = self.mid_block(hidden_states)

        for up_block in self.up_blocks:
            hidden_states = up_block(hidden_states)

        hidden_states = mx.transpose(hidden_states, (0, 2, 3, 1))
        hidden_states = self.conv_norm_out(hidden_states.astype(mx.float32)).astype(ModelConfig.precision)
        hidden_states = nn.silu(hidden_states)
        hidden_states = self.conv_out(hidden_states)
        return mx.transpose(hidden_states, (0, 3, 1, 2))


class Flux2VAE(nn.Module):
    scaling_factor: float = 1.0
    shift_factor: float = 0.0
    latent_channels: int = 32

    def __init__(self):
        super().__init__()
        self.encoder = Flux2Encoder()
        self.decoder = Flux2Decoder()
        self.quant_conv = nn.Conv2d(2 * self.latent_channels, 2 * self.latent_channels, kernel_size=1, padding=0)
        self.post_quant_conv = nn.Conv2d(self.latent_channels, self.latent_channels, kernel_size=1, padding=0)
        self.bn = Flux2BatchNormStats(num_features=4 * self.latent_channels, eps=1e-4, momentum=0.1)

    def encode(self, image: mx.array) -> mx.array:
        if image.ndim == 5:
            image = image[:, :, 0, :, :]
        enc = self.encoder(image)
        enc = mx.transpose(enc, (0, 2, 3, 1))
        enc = self.quant_conv(enc)
        enc = mx.transpose(enc, (0, 3, 1, 2))
        mean, _ = mx.split(enc, 2, axis=1)
        latent = (mean - self.shift_factor) * self.scaling_factor
        return latent

    def decode(self, latents: mx.array) -> mx.array:
        if latents.ndim == 5:
            latents = latents[:, :, 0, :, :]
        latents = (latents / self.scaling_factor) + self.shift_factor
        latents = mx.transpose(latents, (0, 2, 3, 1))
        latents = self.post_quant_conv(latents)
        latents = mx.transpose(latents, (0, 3, 1, 2))
        decoded = self.decoder(latents)
        return decoded

    def decode_packed_latents(self, packed_latents: mx.array) -> mx.array:
        if packed_latents.ndim == 5:
            packed_latents = packed_latents[:, :, 0, :, :]
        bn_mean = self.bn.running_mean.reshape(1, -1, 1, 1)
        bn_std = mx.sqrt(self.bn.running_var.reshape(1, -1, 1, 1) + self.bn.eps)
        latents = packed_latents * bn_std + bn_mean
        latents = self._unpatchify_latents(latents)
        return self.decode(latents)

    @staticmethod
    def _unpatchify_latents(latents: mx.array) -> mx.array:
        batch_size, num_channels, height, width = latents.shape
        latents = mx.reshape(latents, (batch_size, num_channels // 4, 2, 2, height, width))
        latents = mx.transpose(latents, (0, 1, 4, 2, 5, 3))
        latents = mx.reshape(latents, (batch_size, num_channels // 4, height * 2, width * 2))
        return latents
