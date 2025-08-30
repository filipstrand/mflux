import mlx.core as mx
from mlx import nn

from mflux.models.qwen.model.qwen_vae.qwen_image_causal_conv_3d import QwenImageCausalConv3D
from mflux.models.qwen.model.qwen_vae.qwen_image_decoder_3d import QwenImageDecoder3D
from mflux.models.qwen.model.qwen_vae.qwen_image_encoder_3d import QwenImageEncoder3D


class QwenVAE(nn.Module):
    LATENTS_MEAN = mx.array([-0.7571, -0.7089, -0.9113, 0.1075, -0.1745, 0.9653, -0.1517, 1.5508, 0.4134, -0.0715, 0.5517, -0.3632, -0.1922, -0.9497, 0.2503, -0.2921]).reshape(1, 16, 1, 1, 1)
    LATENTS_STD = mx.array([2.8184, 1.4541, 2.3275, 2.6558, 1.2196, 1.7708, 2.6052, 2.0743, 3.2687, 2.1526, 2.8652, 1.5579, 1.6382, 1.1253, 2.8251, 1.916]).reshape(1, 16, 1, 1, 1)

    def __init__(self):
        super().__init__()
        self.decoder = QwenImageDecoder3D()
        self.encoder = QwenImageEncoder3D()
        self.post_quant_conv = QwenImageCausalConv3D(16, 16, 1, 1, 0)
        self.quant_conv = QwenImageCausalConv3D(32, 32, 1, 1, 0)  # Keep 32 channels like diffusers

    def decode(self, latents: mx.array) -> mx.array:
        latents = latents.reshape(latents.shape[0], latents.shape[1], 1, latents.shape[2], latents.shape[3])
        x = self.post_quant_conv(latents)
        decoded = self.decoder(x)
        clamped = mx.minimum(mx.maximum(decoded, -1.0), 1.0)
        return clamped[:, :, 0, :, :]
    
    def encode(self, images: mx.array) -> mx.array:
        images = images.reshape(images.shape[0], images.shape[1], 1, images.shape[2], images.shape[3])
        x = self.encoder.conv_in(images)
        for stage_idx, down_block in enumerate(self.encoder.down_blocks):
            for res_idx, resnet in enumerate(down_block.resnets):
                if stage_idx == 3:
                    residual = x
                    n1 = resnet.norm1(x)
                    a1 = nn.silu(n1)
                    c1 = resnet.conv1(a1)
                    n2 = resnet.norm2(c1)
                    a2 = nn.silu(n2)
                    c2 = resnet.conv2(a2)
                    if resnet.skip_conv is not None:
                        residual = resnet.skip_conv(residual)
                    y = c2 + residual
                    x = y
                else:
                    x = resnet(x)
            if down_block.downsamplers is not None:
                x = down_block.downsamplers[0](x)

        x = self.encoder.mid_block(x)
        norm_in = x
        x = self.encoder.norm_out(norm_in)
        x = nn.silu(x)
        encoded = self.encoder.conv_out(x)
        h = self.quant_conv(encoded)
        mean = h[:, :16, :, :, :]
        mean = mean[:, :, 0, :, :]
        return mean
