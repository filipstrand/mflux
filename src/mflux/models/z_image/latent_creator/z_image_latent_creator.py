import mlx.core as mx

from mflux.models.common.config import ModelConfig


class ZImageLatentCreator:
    @staticmethod
    def create_noise(seed: int, height: int, width: int) -> mx.array:
        return mx.random.normal(
            shape=[
                16,
                1,
                height // 8,
                width // 8,
            ],
            key=mx.random.key(seed),
        ).astype(ModelConfig.precision)

    @staticmethod
    def pack_latents(latents: mx.array, height: int, width: int) -> mx.array:  # noqa: ARG004
        if latents.ndim == 5:
            latents = latents[:, :, 0, :, :]
        latents = mx.expand_dims(latents, axis=2)
        latents = mx.squeeze(latents, axis=0)
        return latents

    @staticmethod
    def unpack_latents(latents: mx.array, height: int, width: int) -> mx.array:  # noqa: ARG004
        latents = mx.expand_dims(latents, axis=0)
        latents = mx.squeeze(latents, axis=2)
        return latents
