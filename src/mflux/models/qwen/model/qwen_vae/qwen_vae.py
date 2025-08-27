import mlx.core as mx
from mlx import nn

from mflux.models.qwen.model.qwen_vae.qwen_image_causal_conv_3d import QwenImageCausalConv3D
from mflux.models.qwen.model.qwen_vae.qwen_image_decoder_3d import QwenImageDecoder3D


class QwenVAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.post_quant_conv = QwenImageCausalConv3D(16, 16, 1, 1, 0)
        self.decoder = QwenImageDecoder3D()

    def decode(self, latents: mx.array) -> mx.array:
        latents = self.post_quant_conv(latents)
        return self.decoder(latents)

    def decode_latents(self, latents: mx.array) -> mx.array:
        latents = latents.reshape(latents.shape[0], latents.shape[1], 1, latents.shape[2], latents.shape[3])
        latents_mean = mx.array(
            [
                -0.7571,
                -0.7089,
                -0.9113,
                0.1075,
                -0.1745,
                0.9653,
                -0.1517,
                1.5508,
                0.4134,
                -0.0715,
                0.5517,
                -0.3632,
                -0.1922,
                -0.9497,
                0.2503,
                -0.2921,
            ]
        ).reshape(1, 16, 1, 1, 1)

        latents_std = mx.array(
            [
                2.8184,
                1.4541,
                2.3275,
                2.6558,
                1.2196,
                1.7708,
                2.6052,
                2.0743,
                3.2687,
                2.1526,
                2.8652,
                1.5579,
                1.6382,
                1.1253,
                2.8251,
                1.916,
            ]
        ).reshape(1, 16, 1, 1, 1)
        latents = latents / (1.0 / latents_std) + latents_mean
        decoded = self.decode(latents)
        return decoded[:, :, 0, :, :]
