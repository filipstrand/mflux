import mlx.core as mx

from mflux.models.common.config import ModelConfig


class ErnieLatentCreator:
    # Patchified latent: [1, 128, H//16, W//16]
    # (VAE 8x encode + 2x2 patchify = 16x total pixel-to-latent scale)
    LATENT_CHANNELS = 128
    SPATIAL_SCALE = 16

    @staticmethod
    def create_noise(seed: int, height: int, width: int) -> mx.array:
        h = height // ErnieLatentCreator.SPATIAL_SCALE
        w = width // ErnieLatentCreator.SPATIAL_SCALE
        return mx.random.normal(
            shape=[1, ErnieLatentCreator.LATENT_CHANNELS, h, w],
            key=mx.random.key(seed),
        ).astype(ModelConfig.precision)

    @staticmethod
    def unpack_latents(latents: mx.array, height: int, width: int) -> mx.array:  # noqa: ARG004
        return latents
