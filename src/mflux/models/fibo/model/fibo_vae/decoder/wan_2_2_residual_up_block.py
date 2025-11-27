import mlx.core as mx
from mlx import nn

from mflux.models.fibo.model.fibo_vae.common.wan_2_2_resample import Wan2_2_Resample
from mflux.models.fibo.model.fibo_vae.common.wan_2_2_residual_block import Wan2_2_ResidualBlock
from mflux.models.fibo.model.fibo_vae.decoder.wan_2_2_dup_up_3d import Wan2_2_DupUp3D


class Wan2_2_ResidualUpBlock(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        num_res_blocks: int,
        temporal_upsample: bool = False,
        up_flag: bool = False,
        non_linearity: str = "silu",
    ):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim

        if up_flag:
            self.avg_shortcut: Wan2_2_DupUp3D | None = Wan2_2_DupUp3D(
                in_channels=in_dim,
                out_channels=out_dim,
                factor_t=2 if temporal_upsample else 1,
                factor_s=2,
            )
        else:
            self.avg_shortcut = None

        resnets: list[Wan2_2_ResidualBlock] = []
        current_dim = in_dim
        for _ in range(num_res_blocks + 1):
            resnets.append(Wan2_2_ResidualBlock(current_dim, out_dim, non_linearity))
            current_dim = out_dim
        self.resnets = resnets

        if up_flag:
            upsample_mode = "upsample3d" if temporal_upsample else "upsample2d"
            self.upsampler: Wan2_2_Resample | None = Wan2_2_Resample(
                out_dim, mode=upsample_mode, upsample_out_dim=out_dim
            )
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
