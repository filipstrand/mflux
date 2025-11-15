import mlx.core as mx
from mlx import nn

from mflux.models.fibo.model.fibo_vae.decoder.wan_causal_conv_3d import WanCausalConv3d
from mflux.models.fibo.model.fibo_vae.decoder.wan_rms_norm import WanRMSNorm


class WanResidualBlock(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        dropout: float = 0.0,
        non_linearity: str = "silu",
    ):
        super().__init__()
        self.norm1 = WanRMSNorm(in_dim, images=False)
        self.conv1 = WanCausalConv3d(in_dim, out_dim, 3, padding=1)
        self.norm2 = WanRMSNorm(out_dim, images=False)
        self.conv2 = WanCausalConv3d(out_dim, out_dim, 3, padding=1)

        if in_dim != out_dim:
            self.conv_shortcut = WanCausalConv3d(in_dim, out_dim, 1, padding=0)
        else:
            self.conv_shortcut = None

    def __call__(self, x: mx.array, resnet_idx: int | None = None, block_idx: int | None = None) -> mx.array:
        h = self.conv_shortcut(x) if self.conv_shortcut is not None else x
        x = self.norm1(x)
        x = nn.silu(x)
        x = self.conv1(x)
        x = self.norm2(x)
        x = nn.silu(x)
        x = self.conv2(x)
        result = x + h
        return result
