import mlx.core as mx

from mflux.models.common.config import ModelConfig
from mflux.models.ideogram4.constants import PATCH_SIZE, VAE_SPATIAL_SCALE
from mflux.models.ideogram4.latent_norm import get_latent_norm


class Ideogram4LatentCreator:
    @staticmethod
    def create_noise(seed: int, height: int, width: int, latent_dim: int = 128) -> mx.array:
        grid_h = height // (PATCH_SIZE * VAE_SPATIAL_SCALE)
        grid_w = width // (PATCH_SIZE * VAE_SPATIAL_SCALE)
        mx.random.seed(seed)
        return mx.random.normal((1, grid_h * grid_w, latent_dim), dtype=mx.float32)

    @staticmethod
    def unpack_latents(latents: mx.array, height: int, width: int) -> mx.array:
        batch_size = latents.shape[0]
        grid_h = height // (PATCH_SIZE * VAE_SPATIAL_SCALE)
        grid_w = width // (PATCH_SIZE * VAE_SPATIAL_SCALE)
        shift, scale = get_latent_norm()
        latents = latents * scale.astype(latents.dtype)[None, None, :] + shift.astype(latents.dtype)[None, None, :]
        ae_channels = latents.shape[-1] // (PATCH_SIZE * PATCH_SIZE)
        latents = latents.reshape(batch_size, grid_h, grid_w, PATCH_SIZE, PATCH_SIZE, ae_channels)
        latents = latents.transpose(0, 5, 1, 3, 2, 4)
        latents = latents.reshape(batch_size, ae_channels, grid_h * PATCH_SIZE, grid_w * PATCH_SIZE)
        return latents.astype(ModelConfig.precision)
