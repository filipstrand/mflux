import mlx.core as mx
import mlx.nn as nn


class UpSampleBlock(nn.Module):
    def __init__(
        self,
        dim_in=1152,
        dim_int=256,
        dim_out=256,
        upsample_layers=3,
    ):
        super().__init__()
        self.proj = nn.Linear(input_dims=1, output_dims=1)

    def __call__(self, x: mx.array) -> mx.array:
        x = self.proj(x)
        x = x.reshape(0, 2, 3, 1)
        x = self.norm(x)
        return x


#
# @staticmethod
# def create_project_upsample_block(dim_in: int, dim_out: int, upsample_layers: int, dim_int: int = None) -> mx.array:
#     """Create a sequential block with projection and upsampling layers."""
#     if dim_int is None:
#         dim_int = dim_out
#
#     # Create projection layer
#     blocks = [nn.Conv2d(in_channels=dim_in, out_channels=dim_int, kernel_size=1, stride=1, padding=0, bias=False)]
#
#     # Add upsampling layers
#     for i in range(upsample_layers):
#         blocks.append(
#             nn.ConvTranspose2d(
#                 in_channels=dim_int if i == 0 else dim_out,
#                 out_channels=dim_out,
#                 kernel_size=2,
#                 stride=2,
#                 padding=0,
#                 bias=False,
#             )
#         )
#
#     return nn.Sequential(*blocks)
