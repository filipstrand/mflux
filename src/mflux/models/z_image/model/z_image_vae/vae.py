import mlx.core as mx
from mlx import nn

from mflux.models.z_image.model.z_image_vae.decoder.decoder import Decoder
from mflux.models.z_image.model.z_image_vae.encoder.encoder import Encoder


class VAE(nn.Module):
    scaling_factor: float = 0.3611
    shift_factor: float = 0.1159

    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def encode(self, image: mx.array) -> mx.array:
        h = self.encoder(image)
        mean, _ = mx.split(h, 2, axis=1)
        return (mean - self.shift_factor) * self.scaling_factor

    def decode(self, latents: mx.array) -> mx.array:
        scaled_latents = (latents / self.scaling_factor) + self.shift_factor
        return self.decoder(scaled_latents)
