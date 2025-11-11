import mlx.core as mx
from mlx import nn

from mflux.models.fibo.model.fibo_vae.common.wan_2_2_attention_block import Wan2_2_AttentionBlock
from mflux.models.fibo.model.fibo_vae.common.wan_2_2_resample import Wan2_2_Resample
from mflux.models.fibo.model.fibo_vae.common.wan_2_2_residual_block import Wan2_2_ResidualBlock
from mflux.models.fibo.model.fibo_vae.encoder.wan_2_2_avg_down_3d import Wan2_2_AvgDown3D


class Wan2_2_DownBlock(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        num_res_blocks: int,
        attn_scales: list[float] | None = None,
        scale: float = 1.0,
        temporal_downsample: bool = False,
        non_linearity: str = "silu",
        is_last: bool = False,
    ):
        super().__init__()
        if attn_scales is None:
            attn_scales = []

        resnets: list[nn.Module] = []
        current_dim = in_dim
        for _ in range(num_res_blocks):
            resnets.append(Wan2_2_ResidualBlock(current_dim, out_dim, non_linearity))
            if scale in attn_scales:
                resnets.append(Wan2_2_AttentionBlock(out_dim))
            current_dim = out_dim

        self.resnets = resnets

        # Shortcut path with downsample (mirrors AvgDown3D in diffusers)
        self.avg_shortcut = Wan2_2_AvgDown3D(
            in_dim,
            out_dim,
            factor_t=2 if temporal_downsample else 1,
            factor_s=2 if not is_last else 1,
        )

        # Main path downsampler
        if not is_last:
            mode = "downsample3d" if temporal_downsample else "downsample2d"
            self.downsampler = Wan2_2_Resample(out_dim, mode=mode)
        else:
            self.downsampler = None

    def __call__(self, x: mx.array) -> mx.array:
        x_copy = x
        for layer in self.resnets:
            x = layer(x)
        if self.downsampler is not None:
            x = self.downsampler(x)
        return x + self.avg_shortcut(x_copy)
