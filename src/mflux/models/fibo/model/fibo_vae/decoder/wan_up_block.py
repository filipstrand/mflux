"""Wan Up Blocks for FIBO VAE decoder.

This module contains:
- WanUpBlock: simple upsampling block (non-residual wrapper)
- DupUp3D: channel/time/space duplication shortcut used in residual up blocks
- WanResidualUpBlock: residual upsampling block matching diffusers WanResidualUpBlock
"""

import mlx.core as mx
from mlx import nn

from mflux.models.fibo.model.fibo_vae.decoder.wan_resample import WanResample
from mflux.models.fibo.model.fibo_vae.decoder.wan_residual_block import WanResidualBlock


class WanUpBlock(nn.Module):
    """Simple upsampling block for decoder (non-residual wrapper)."""

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        num_res_blocks: int,
        dropout: float = 0.0,
        upsample_mode: str | None = None,
        non_linearity: str = "silu",
    ):
        super().__init__()
        # Create residual blocks
        self.resnets: list[WanResidualBlock] = []
        current_dim = in_dim
        for _ in range(num_res_blocks + 1):
            self.resnets.append(WanResidualBlock(current_dim, out_dim, dropout, non_linearity))
            current_dim = out_dim

        # Add upsampling if needed
        self.upsampler: WanResample | None = None
        if upsample_mode is not None:
            # Match diffusers: WanResample(out_dim, mode=upsample_mode, upsample_out_dim=out_dim)
            self.upsampler = WanResample(out_dim, mode=upsample_mode, upsample_out_dim=out_dim)

    def __call__(self, x: mx.array, block_idx: int | None = None) -> mx.array:
        # Apply residual blocks
        for i, resnet in enumerate(self.resnets):
            x = resnet(x, resnet_idx=i, block_idx=block_idx)

        # Apply upsampling if present
        if self.upsampler is not None:
            x = self.upsampler(x, block_idx=block_idx)

        return x


class DupUp3D(nn.Module):
    """MLX implementation of diffusers DupUp3D for the residual shortcut."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        factor_t: int,
        factor_s: int = 1,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.factor_t = factor_t
        self.factor_s = factor_s
        self.factor = self.factor_t * self.factor_s * self.factor_s

        assert out_channels * self.factor % in_channels == 0
        self.repeats = out_channels * self.factor // in_channels

    def __call__(self, x: mx.array, first_chunk: bool = False) -> mx.array:
        # x: (batch, channels, time, height, width)
        b, c, t, h, w = x.shape

        # Repeat channels
        x = mx.repeat(x, self.repeats, axis=1)  # (b, c*repeats, t, h, w)

        # Reshape into (b, out_channels, factor_t, factor_s, factor_s, t, h, w)
        x = mx.reshape(
            x,
            (
                b,
                self.out_channels,
                self.factor_t,
                self.factor_s,
                self.factor_s,
                t,
                h,
                w,
            ),
        )

        # Permute to interleave the upsample dimensions:
        # (b, out_c, t, factor_t, h, factor_s, w, factor_s)
        x = mx.transpose(x, (0, 1, 5, 2, 6, 3, 7, 4))

        # Merge upsample factors into time/height/width
        x = mx.reshape(
            x,
            (
                b,
                self.out_channels,
                t * self.factor_t,
                h * self.factor_s,
                w * self.factor_s,
            ),
        )

        # For first chunk with temporal upsample, drop the initial frames
        if first_chunk and self.factor_t > 1:
            x = x[:, :, self.factor_t - 1 :, :, :]

        return x


class WanResidualUpBlock(nn.Module):
    """Residual upsampling block matching diffusers WanResidualUpBlock."""

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        num_res_blocks: int,
        dropout: float = 0.0,
        temporal_upsample: bool = False,
        up_flag: bool = False,
        non_linearity: str = "silu",
    ):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim

        # Outer shortcut path
        if up_flag:
            self.avg_shortcut: DupUp3D | None = DupUp3D(
                in_channels=in_dim,
                out_channels=out_dim,
                factor_t=2 if temporal_upsample else 1,
                factor_s=2,
            )
        else:
            self.avg_shortcut = None

        # Residual blocks
        resnets: list[WanResidualBlock] = []
        current_dim = in_dim
        for _ in range(num_res_blocks + 1):
            resnets.append(WanResidualBlock(current_dim, out_dim, dropout, non_linearity))
            current_dim = out_dim
        self.resnets = resnets

        # Upsampler
        if up_flag:
            upsample_mode = "upsample3d" if temporal_upsample else "upsample2d"
            self.upsampler: WanResample | None = WanResample(out_dim, mode=upsample_mode, upsample_out_dim=out_dim)
        else:
            self.upsampler = None

    def __call__(self, x: mx.array, block_idx: int | None = None, first_chunk: bool = False) -> mx.array:
        x_copy = x

        # Inner residual stack
        for i, resnet in enumerate(self.resnets):
            x = resnet(x, resnet_idx=i, block_idx=block_idx)

        # Upsample main path
        if self.upsampler is not None:
            x = self.upsampler(x, block_idx=block_idx)

        # Add outer shortcut if present
        if self.avg_shortcut is not None:
            x = x + self.avg_shortcut(x_copy, first_chunk=first_chunk)

        return x
