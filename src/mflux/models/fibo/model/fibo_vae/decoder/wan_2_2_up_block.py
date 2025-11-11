import mlx.core as mx
from mlx import nn

from mflux.models.fibo.model.fibo_vae.common.wan_2_2_resample import Wan2_2_Resample
from mflux.models.fibo.model.fibo_vae.common.wan_2_2_residual_block import Wan2_2_ResidualBlock


class Wan2_2_UpBlock(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        num_res_blocks: int,
        upsample_mode: str | None = None,
        non_linearity: str = "silu",
    ):
        super().__init__()
        self.resnets: list[Wan2_2_ResidualBlock] = []
        current_dim = in_dim
        for _ in range(num_res_blocks + 1):
            self.resnets.append(Wan2_2_ResidualBlock(current_dim, out_dim, non_linearity))
            current_dim = out_dim
        self.upsampler: Wan2_2_Resample | None = None
        if upsample_mode is not None:
            self.upsampler = Wan2_2_Resample(out_dim, mode=upsample_mode, upsample_out_dim=out_dim)

    def __call__(self, x: mx.array, block_idx: int | None = None) -> mx.array:
        for i, resnet in enumerate(self.resnets):
            x = resnet(x, resnet_idx=i, block_idx=block_idx)

        if self.upsampler is not None:
            x = self.upsampler(x, block_idx=block_idx)

        return x
