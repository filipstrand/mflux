import mlx.core as mx
from mlx import nn

from mflux.models.fibo.model.fibo_vae.decoder.wan_attention_block import WanAttentionBlock
from mflux.models.fibo.model.fibo_vae.decoder.wan_causal_conv_3d import WanCausalConv3d
from mflux.models.fibo.model.fibo_vae.decoder.wan_mid_block import WanMidBlock
from mflux.models.fibo.model.fibo_vae.decoder.wan_residual_block import WanResidualBlock
from mflux.models.fibo.model.fibo_vae.decoder.wan_rms_norm import WanRMSNorm


class WanDownBlock(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        num_res_blocks: int,
        dropout: float = 0.0,
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
            resnets.append(WanResidualBlock(current_dim, out_dim, dropout, non_linearity))
            if scale in attn_scales:
                resnets.append(WanAttentionBlock(out_dim))
            current_dim = out_dim

        self.resnets = resnets

        if not is_last:
            mode = "downsample3d" if temporal_downsample else "downsample2d"
            # WanResample is imported lazily to avoid circular import at module load time
            from mflux.models.fibo.model.fibo_vae.decoder.wan_resample import WanResample

            self.downsampler = WanResample(out_dim, mode=mode)
        else:
            self.downsampler = None

    def __call__(self, x: mx.array) -> mx.array:
        for layer in self.resnets:
            x = layer(x)
        if self.downsampler is not None:
            x = self.downsampler(x)
        return x


class WanEncoder3d(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        dim: int = 128,
        z_dim: int = 4,
        dim_mult: list[int] | None = None,
        num_res_blocks: int = 2,
        attn_scales: list[float] | None = None,
        temporal_downsample: list[bool] | None = None,
        dropout: float = 0.0,
        non_linearity: str = "silu",
        is_residual: bool = False,
    ):
        super().__init__()

        if dim_mult is None:
            dim_mult = [1, 2, 4, 4]
        if attn_scales is None:
            attn_scales = []
        if temporal_downsample is None:
            temporal_downsample = [False, True, True]

        if is_residual:
            raise NotImplementedError("Residual down blocks are not implemented for WanEncoder3d in MLX.")

        self.dim = dim
        self.z_dim = z_dim
        self.dim_mult = dim_mult
        self.num_res_blocks = num_res_blocks
        self.attn_scales = attn_scales
        dims = [dim * u for u in [1] + dim_mult]
        self.temporal_downsample = temporal_downsample

        self.conv_in = WanCausalConv3d(in_channels, dims[0], 3, padding=1)
        scale = 1.0

        self.down_blocks: list[WanDownBlock] = []
        for i, (in_dim, out_dim) in enumerate(zip(dims[:-1], dims[1:])):
            block = WanDownBlock(
                in_dim=in_dim,
                out_dim=out_dim,
                num_res_blocks=num_res_blocks,
                dropout=dropout,
                attn_scales=attn_scales,
                scale=scale,
                temporal_downsample=temporal_downsample[i] if i < len(temporal_downsample) else False,
                non_linearity=non_linearity,
                is_last=i == len(dim_mult) - 1,
            )
            self.down_blocks.append(block)
            if i != len(dim_mult) - 1:
                scale /= 2.0

        self.mid_block = WanMidBlock(out_dim, dropout, non_linearity, num_layers=1)

        self.norm_out = WanRMSNorm(out_dim, images=False)
        self.conv_out = WanCausalConv3d(out_dim, z_dim, 3, padding=1)

    def __call__(self, x: mx.array) -> mx.array:
        x = self.conv_in(x)
        for block in self.down_blocks:
            x = block(x)
        x = self.mid_block(x)
        x = self.norm_out(x)
        x = nn.silu(x)
        x = self.conv_out(x)
        return x
