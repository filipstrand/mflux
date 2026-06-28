import mlx.core as mx

VAE_SCALE_FACTOR = 8
PATCH_SIZE = 2
LATENT_CHANNELS = 16


class Krea2LatentCreator:
    """Latent packing / position ids for Krea 2 (matches diffusers Krea2Pipeline)."""

    @staticmethod
    def create_packed_noise(seed: int, height: int, width: int, batch_size: int = 1) -> mx.array:
        latent_h = height // VAE_SCALE_FACTOR
        latent_w = width // VAE_SCALE_FACTOR
        mx.random.seed(seed)
        latents = mx.random.normal(
            shape=(batch_size, LATENT_CHANNELS, latent_h, latent_w),
            dtype=mx.float32,
        )
        return Krea2LatentCreator.pack_latents(latents, batch_size, LATENT_CHANNELS, latent_h, latent_w)

    @staticmethod
    def pack_latents(latents: mx.array, batch_size: int, num_channels: int, height: int, width: int) -> mx.array:
        p = PATCH_SIZE
        latents = latents.reshape(batch_size, num_channels, height // p, p, width // p, p)
        latents = latents.transpose(0, 2, 4, 1, 3, 5)
        latents = latents.reshape(batch_size, (height // p) * (width // p), num_channels * p * p)
        return latents

    @staticmethod
    def unpack_latents(latents: mx.array, height: int, width: int) -> mx.array:
        batch_size, _, channels = latents.shape
        p = PATCH_SIZE
        h = p * (int(height) // (VAE_SCALE_FACTOR * p))
        w = p * (int(width) // (VAE_SCALE_FACTOR * p))
        latents = latents.reshape(batch_size, h // p, w // p, channels // (p * p), p, p)
        latents = latents.transpose(0, 3, 1, 4, 2, 5)
        latents = latents.reshape(batch_size, channels // (p * p), 1, h, w)
        return latents

    @staticmethod
    def prepare_position_ids(text_seq_len: int, grid_height: int, grid_width: int) -> mx.array:
        text_ids = mx.zeros((text_seq_len, 3), dtype=mx.float32)
        h_coords = mx.broadcast_to(mx.arange(grid_height)[:, None], (grid_height, grid_width))
        w_coords = mx.broadcast_to(mx.arange(grid_width)[None, :], (grid_height, grid_width))
        t_coords = mx.zeros((grid_height, grid_width), dtype=mx.float32)
        image_ids = mx.stack([t_coords, h_coords.astype(mx.float32), w_coords.astype(mx.float32)], axis=-1)
        image_ids = image_ids.reshape(grid_height * grid_width, 3)
        return mx.concatenate([text_ids, image_ids], axis=0)
