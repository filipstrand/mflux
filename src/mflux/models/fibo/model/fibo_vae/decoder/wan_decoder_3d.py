import mlx.core as mx
from mlx import nn

from mflux.models.fibo.model.fibo_vae.decoder.wan_causal_conv_3d import WanCausalConv3d
from mflux.models.fibo.model.fibo_vae.decoder.wan_mid_block import WanMidBlock
from mflux.models.fibo.model.fibo_vae.decoder.wan_residual_up_block import WanResidualUpBlock
from mflux.models.fibo.model.fibo_vae.decoder.wan_rms_norm import WanRMSNorm


class WanDecoder3d(nn.Module):
    def __init__(
        self,
        dim: int = 256,
        z_dim: int = 48,
        dim_mult: list[int] = [1, 2, 4, 4],
        num_res_blocks: int = 2,
        temporal_upsample: list[bool] | None = None,
        dropout: float = 0.0,
        non_linearity: str = "silu",
        out_channels: int = 12,
    ):
        super().__init__()
        self.dim = dim
        self.z_dim = z_dim
        self.dim_mult = dim_mult
        self.num_res_blocks = num_res_blocks
        self.temporal_upsample = temporal_upsample or []
        dims = [dim * u for u in [dim_mult[-1]] + dim_mult[::-1]]
        self.conv_in = WanCausalConv3d(z_dim, dims[0], 3, padding=1, name="decoder_conv_in")
        self.mid_block = WanMidBlock(dims[0], dropout, non_linearity, num_layers=1)
        self.up_blocks: list[WanResidualUpBlock] = []
        for i, (in_dim, out_dim) in enumerate(zip(dims[:-1], dims[1:])):
            up_flag = i != len(dim_mult) - 1
            temporal_flag = (
                bool(self.temporal_upsample and i < len(self.temporal_upsample) and self.temporal_upsample[i])
                if up_flag
                else False
            )
            up_block = WanResidualUpBlock(
                in_dim=in_dim,
                out_dim=out_dim,
                num_res_blocks=num_res_blocks,
                dropout=dropout,
                temporal_upsample=temporal_flag,
                up_flag=up_flag,
                non_linearity=non_linearity,
            )
            self.up_blocks.append(up_block)
        self.norm_out = WanRMSNorm(out_dim, images=False)
        self.conv_out = WanCausalConv3d(out_dim, out_channels, 3, padding=1, name="decoder_conv_out")

    def __call__(self, x: mx.array) -> mx.array:
        x = self.conv_in(x)
        x = self.mid_block(x)
        for i, up_block in enumerate(self.up_blocks):
            x = up_block(x, block_idx=i, first_chunk=True)
        x = self.norm_out(x)
        x = nn.silu(x)
        x = self.conv_out(x)
        return x
