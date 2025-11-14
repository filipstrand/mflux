import mlx.core as mx
from mlx import nn

from mflux.models.fibo.model.fibo_vae.decoder.wan_resample import WanResample
from mflux.models.fibo.model.fibo_vae.decoder.wan_residual_block import WanResidualBlock


class WanUpBlock(nn.Module):
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
        self.resnets: list[WanResidualBlock] = []
        current_dim = in_dim
        for _ in range(num_res_blocks + 1):
            self.resnets.append(WanResidualBlock(current_dim, out_dim, dropout, non_linearity))
            current_dim = out_dim
        self.upsampler: WanResample | None = None
        if upsample_mode is not None:
            self.upsampler = WanResample(out_dim, mode=upsample_mode, upsample_out_dim=out_dim)

    def __call__(self, x: mx.array, block_idx: int | None = None) -> mx.array:
        for i, resnet in enumerate(self.resnets):
            x = resnet(x, resnet_idx=i, block_idx=block_idx)

        if self.upsampler is not None:
            x = self.upsampler(x, block_idx=block_idx)

        return x


class DupUp3D(nn.Module):
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
        self.repeats = out_channels * self.factor // in_channels

    def __call__(self, x: mx.array, first_chunk: bool = False) -> mx.array:
        b, c, t, h, w = x.shape
        x = mx.repeat(x, self.repeats, axis=1)  # (b, c*repeats, t, h, w)
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

        x = mx.transpose(x, (0, 1, 5, 2, 6, 3, 7, 4))
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

        if first_chunk and self.factor_t > 1:
            x = x[:, :, self.factor_t - 1 :, :, :]

        return x


class WanResidualUpBlock(nn.Module):
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

        for i, resnet in enumerate(self.resnets):
            x = resnet(x, resnet_idx=i, block_idx=block_idx)

        if self.upsampler is not None:
            x = self.upsampler(x, block_idx=block_idx)

        if self.avg_shortcut is not None:
            x = x + self.avg_shortcut(x_copy, first_chunk=first_chunk)

        return x
