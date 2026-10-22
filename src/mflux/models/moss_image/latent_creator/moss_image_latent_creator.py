import mlx.core as mx


class MOSSImageLatentCreator:
    LATENT_CHANNELS = 128
    SPATIAL_SCALE = 16

    @staticmethod
    def create_noise(seed: int, height: int, width: int) -> mx.array:
        # The reference samples the initial noise in float32 and keeps the latents in float32.
        MOSSImageLatentCreator.check_size(height=height, width=width)
        shape = (
            1,
            MOSSImageLatentCreator.LATENT_CHANNELS,
            height // MOSSImageLatentCreator.SPATIAL_SCALE,
            width // MOSSImageLatentCreator.SPATIAL_SCALE,
        )
        return mx.random.normal(shape=shape, key=mx.random.key(seed), dtype=mx.float32)

    @staticmethod
    def check_size(height: int, width: int) -> None:
        scale = MOSSImageLatentCreator.SPATIAL_SCALE
        if height <= 0 or width <= 0 or height % scale or width % scale:
            raise ValueError(f"MOSS-Image needs a width and height that are positive multiples of 16, got {width}x{height}.")  # fmt: off

    @staticmethod
    def unpack_latents(latents: mx.array, height: int, width: int) -> mx.array:  # noqa: ARG004
        # The latents are already the packed 128-channel FLUX.2 VAE layout.
        return latents
