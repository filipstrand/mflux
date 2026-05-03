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
    def pack_latents(latents: mx.array, height: int, width: int) -> mx.array:  # noqa: ARG004
        # Input: [B, 32, H//8, W//8] from VAE encode
        B, C, H, W = latents.shape
        h, w = H // 2, W // 2  # → H//16, W//16

        # 2×2 spatial patchification (inverse of Flux2VAE._unpatchify_latents)
        latents = latents.reshape(B, C, h, 2, w, 2)   # [B, 32, h, 2, w, 2]
        latents = latents.transpose(0, 1, 3, 5, 2, 4)  # [B, 32, 2, 2, h, w]
        return latents.reshape(B, C * 4, h, w)          # [B, 128, H//16, W//16]

    @staticmethod
    def bn_normalize_latents(latents: mx.array, vae) -> mx.array:
        bn_mean = vae.bn.running_mean.reshape(1, -1, 1, 1)
        bn_std = mx.sqrt(vae.bn.running_var.reshape(1, -1, 1, 1) + vae.bn.eps)
        return (latents - bn_mean) / bn_std

    @staticmethod
    def unpack_latents(latents: mx.array, height: int, width: int) -> mx.array:  # noqa: ARG004
        return latents
