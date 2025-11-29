import mlx.core as mx

from mflux.models.flux.latent_creator.flux_latent_creator import FluxLatentCreator


class QwenLatentCreator:
    @staticmethod
    def create_noise(seed: int, height: int, width: int) -> mx.array:
        return FluxLatentCreator.create_noise(seed, height, width)

    @staticmethod
    def pack_latents(latents: mx.array, height: int, width: int, num_channels_latents: int = 16) -> mx.array:
        return FluxLatentCreator.pack_latents(latents, height, width, num_channels_latents)

    @staticmethod
    def unpack_latents(latents: mx.array, height: int, width: int) -> mx.array:
        return FluxLatentCreator.unpack_latents(latents, height, width)
