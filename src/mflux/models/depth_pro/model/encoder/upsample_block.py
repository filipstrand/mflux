import mlx.core as mx
import mlx.nn as nn

from mflux.models.depth_pro.model.depth_pro_util import DepthProUtil


class UpSampleBlock(nn.Module):
    def __init__(
        self,
        dim_in: int = 1152,
        dim_int: int = 256,
        dim_out: int = 256,
        upsample_layers: int = 3,
    ):
        super().__init__()
        self.layers = UpSampleBlock._create_layers(
            dim_in=dim_in,
            dim_int=dim_int,
            dim_out=dim_out,
            upsample_layers=upsample_layers
        )  # fmt: off

    @staticmethod
    def _create_layers(dim_in: int, dim_out: int, upsample_layers: int, dim_int: int | None = None) -> list[nn.Module]:
        if dim_int is None:
            dim_int = dim_out

        # Create projection layer
        layers = [nn.Conv2d(in_channels=dim_in, out_channels=dim_int, kernel_size=1, stride=1, padding=0, bias=False)]

        # Add upsampling layers
        layers.extend(
            [
                nn.ConvTranspose2d(
                    in_channels=dim_int if i == 0 else dim_out,
                    out_channels=dim_out,
                    kernel_size=2,
                    stride=2,
                    padding=0,
                    bias=False,
                )
                for i in range(upsample_layers)
            ]
        )

        return layers

    def __call__(self, x: mx.array) -> mx.array:
        for layer in self.layers:
            x = DepthProUtil.apply_conv(x, layer)
        return x
