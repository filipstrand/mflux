import mlx.core as mx
from mlx import nn

from mflux.models.flux2.model.flux2_vae.common.batch_norm_stats import Flux2BatchNormStats
from mflux.models.flux2.model.flux2_vae.decoder.decoder import Flux2Decoder
from mflux.models.flux2.model.flux2_vae.encoder.encoder import Flux2Encoder


class Flux2VAE(nn.Module):
    scaling_factor: float = 1.0
    shift_factor: float = 0.0
    latent_channels: int = 32

    def __init__(self):
        super().__init__()
        self.encoder = Flux2Encoder()
        self.decoder = Flux2Decoder()
        self.quant_conv = nn.Conv2d(2 * self.latent_channels, 2 * self.latent_channels, kernel_size=1, padding=0)
        self.post_quant_conv = nn.Conv2d(self.latent_channels, self.latent_channels, kernel_size=1, padding=0)
        self.bn = Flux2BatchNormStats(num_features=4 * self.latent_channels, eps=1e-4, momentum=0.1)

    def encode(self, image: mx.array) -> mx.array:
        if image.ndim == 5:
            image = image[:, :, 0, :, :]
        enc = self.encoder(image)
        enc = mx.transpose(enc, (0, 2, 3, 1))
        enc = self.quant_conv(enc)
        enc = mx.transpose(enc, (0, 3, 1, 2))
        mean, _ = mx.split(enc, 2, axis=1)
        latent = (mean - self.shift_factor) * self.scaling_factor
        return latent

    def decode(self, latents: mx.array) -> mx.array:
        if latents.ndim == 5:
            latents = latents[:, :, 0, :, :]
        latents = (latents / self.scaling_factor) + self.shift_factor
        latents = mx.transpose(latents, (0, 2, 3, 1))
        latents = self.post_quant_conv(latents)
        latents = mx.transpose(latents, (0, 3, 1, 2))
        decoded = self.decoder(latents)
        return decoded

    def decode_packed_latents(self, packed_latents: mx.array) -> mx.array:
        if packed_latents.ndim == 5:
            packed_latents = packed_latents[:, :, 0, :, :]
        bn_mean = self.bn.running_mean.reshape(1, -1, 1, 1)
        bn_std = mx.sqrt(self.bn.running_var.reshape(1, -1, 1, 1) + self.bn.eps)
        latents = packed_latents * bn_std + bn_mean
        latents = self._unpatchify_latents(latents)
        return self.decode(latents)

    @staticmethod
    def _unpatchify_latents(latents: mx.array) -> mx.array:
        batch_size, num_channels, height, width = latents.shape
        latents = mx.reshape(latents, (batch_size, num_channels // 4, 2, 2, height, width))
        latents = mx.transpose(latents, (0, 1, 4, 2, 5, 3))
        latents = mx.reshape(latents, (batch_size, num_channels // 4, height * 2, width * 2))
        return latents
