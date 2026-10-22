import mlx.core as mx
from mlx import nn

from mflux.models.common.vae.block_decoder import VAEBlockDecoder
from mflux.models.z_image.model.z_image_vae.decoder.decoder import Decoder
from mflux.models.z_image.model.z_image_vae.encoder.encoder import Encoder


class VAE(nn.Module):
    scaling_factor: float = 0.3611
    shift_factor: float = 0.1159
    spatial_scale = 8
    latent_channels = 16
    pid_variant = "flux"  # Z-Image shares Flux's 16ch/8x latent space; nvidia/PiD ships no z-image checkpoint
    # See VAEBlockDecoder. The resnets of this decoder run GroupNorm in the input dtype, not in fp32.
    decode_mode: str = VAEBlockDecoder.LEGACY

    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def encode(self, image: mx.array) -> mx.array:
        if image.ndim == 5:
            image = image[:, :, 0, :, :]
        h = self.encoder(image)
        mean, _ = mx.split(h, 2, axis=1)
        latent = (mean - self.shift_factor) * self.scaling_factor
        return latent[:, :, None, :, :]

    def decode(self, latents: mx.array) -> mx.array:
        if latents.ndim == 5:
            latents = latents[:, :, 0, :, :]
        if self.decode_mode != VAEBlockDecoder.LEGACY:
            strips = self.decode_mode == VAEBlockDecoder.LOW_MEMORY
            return VAEBlockDecoder.decode(vae=self, latent=latents, strips=strips, resnet_norm_fp32=False)[
                :, :, None, :, :
            ]
        scaled_latents = (latents / self.scaling_factor) + self.shift_factor
        decoded = self.decoder(scaled_latents)
        return decoded[:, :, None, :, :]
