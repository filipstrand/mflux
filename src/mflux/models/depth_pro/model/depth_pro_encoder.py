import math

import mlx.core as mx
import mlx.nn as nn

from mflux.models.depth_pro.conv_utils import ConvUtils
from mflux.models.depth_pro.dino_v2.dino_vision_transformer import DinoVisionTransformer
from mflux.models.depth_pro.upsample_block import UpSampleBlock


class DepthProEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.patch_encoder = DinoVisionTransformer()
        self.image_encoder = DinoVisionTransformer()
        self.upsample_latent0 = UpSampleBlock(dim_in=1024, dim_int=256, dim_out=256, upsample_layers=3)
        self.upsample_latent1 = UpSampleBlock(dim_in=1024, dim_out=256, upsample_layers=2)
        self.upsample0 = UpSampleBlock(dim_in=1024, dim_out=512, upsample_layers=1)
        self.upsample1 = UpSampleBlock(dim_in=1024, dim_out=1024, upsample_layers=1)
        self.upsample2 = UpSampleBlock(dim_in=1024, dim_out=1024, upsample_layers=1)
        self.upsample_lowres = nn.ConvTranspose2d(in_channels=1024, out_channels=1024, kernel_size=2, stride=2, padding=0, bias=True)  # fmt: off
        self.fuse_lowres = nn.Conv2d(in_channels=1024 * 2, out_channels=1024, kernel_size=1, stride=1, padding=0, bias=True)  # fmt: off

    def __call__(
        self,
        x0: mx.array,
        x1: mx.array,
        x2: mx.array,
    ) -> tuple[mx.array, mx.array, mx.array, mx.array, mx.array]:
        # 1: Run the backbone patch encoder model
        x_pyramid_patches = mx.concatenate((x0, x1, x2), axis=0)
        x_pyramid_encodings, backbone_highres_hook0, backbone_highres_hook1 = self.patch_encoder(x_pyramid_patches)
        x_pyramid_encodings = DepthProEncoder._reshape_feature(x_pyramid_encodings, width=24, height=24)
        x_latent0_encodings = DepthProEncoder._reshape_feature(backbone_highres_hook0, width=24, height=24)
        x_latent1_encodings = DepthProEncoder._reshape_feature(backbone_highres_hook1, width=24, height=24)

        # Calculate indices for splitting
        x0_encodings = x_pyramid_encodings[: len(x0)]
        x1_encodings = x_pyramid_encodings[len(x0) : len(x0) + len(x1)]
        x2_encodings = x_pyramid_encodings[len(x0) + len(x1) :]

        # 2. Merging
        x_latent0_features = DepthProEncoder._merge(x_latent0_encodings[: 1 * 5 * 5], batch_size=1, padding=3)
        x_latent1_features = DepthProEncoder._merge(x_latent1_encodings[: 1 * 5 * 5], batch_size=1, padding=3)
        x0_features = DepthProEncoder._merge(x0_encodings, batch_size=1, padding=3)
        x1_features = DepthProEncoder._merge(x1_encodings, batch_size=1, padding=6)
        x2_features = x2_encodings

        # 3. Upsample feature maps.
        x_latent0_features = self.upsample_latent0(x_latent0_features)
        x_latent1_features = self.upsample_latent1(x_latent1_features)
        x0_features = self.upsample0(x0_features)
        x1_features = self.upsample1(x1_features)
        x2_features = self.upsample2(x2_features)

        # 4. Apply the image encoder model.
        x_global_features, _, _ = self.image_encoder(x2)
        x_global_features = DepthProEncoder._reshape_feature(embeddings=x_global_features, width=24, height=24)
        x_global_features = ConvUtils.apply_conv(x_global_features, self.upsample_lowres)
        x_global_features = mx.concatenate((x2_features, x_global_features), axis=1)
        x_global_features = ConvUtils.apply_conv(x_global_features, self.fuse_lowres)

        return (
            x_latent0_features,
            x_latent1_features,
            x0_features,
            x1_features,
            x_global_features,
        )

    @staticmethod
    def _reshape_feature(
        embeddings: mx.array,
        width: int,
        height: int,
        cls_token_offset: int = 1,
    ) -> mx.array:
        b, hw, c = embeddings.shape
        if cls_token_offset > 0:
            embeddings = embeddings[:, cls_token_offset:, :]
        embeddings = embeddings.reshape(b, height, width, c).transpose(0, 3, 1, 2)
        return embeddings

    @staticmethod
    def _merge(x: mx.array, batch_size: int, padding: int = 3) -> mx.array:
        steps = int(math.sqrt(x.shape[0] // batch_size))

        idx = 0

        output_list = []
        for j in range(steps):
            output_row_list = []
            for i in range(steps):
                output = x[batch_size * idx : batch_size * (idx + 1)]

                if j != 0:
                    output = output[..., padding:, :]
                if i != 0:
                    output = output[..., :, padding:]
                if j != steps - 1:
                    output = output[..., :-padding, :]
                if i != steps - 1:
                    output = output[..., :, :-padding]

                output_row_list.append(output)
                idx += 1

            output_row = mx.concatenate(output_row_list, axis=-1)
            output_list.append(output_row)
        output = mx.concatenate(output_list, axis=-2)
        return output
