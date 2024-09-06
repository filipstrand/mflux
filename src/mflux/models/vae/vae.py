import mlx.core as mx
from mlx import nn

from mflux.models.vae.decoder.decoder import Decoder
from mflux.models.vae.encoder.encoder import Encoder


class VAE(nn.Module):
    scaling_factor: int = 0.3611
    shift_factor: int = 0.1159

    def __init__(self):
        super().__init__()
        self.decoder = Decoder()
        self.encoder = Encoder()

    def decode(self, latents: mx.array) -> mx.array:
        scaled_latents = (latents / self.scaling_factor) + self.shift_factor
        return self.decoder.decode(scaled_latents)

    def encode(self, latents: mx.array) -> mx.array:
        latents = self.encoder.encode(latents)
        mean, _ = mx.split(latents, 2, axis=1)
        return (mean - self.shift_factor) * self.scaling_factor
