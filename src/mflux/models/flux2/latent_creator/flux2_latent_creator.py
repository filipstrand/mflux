import mlx.core as mx

from mflux.models.common.config.model_config import ModelConfig


class Flux2LatentCreator:
    @staticmethod
    def pack_latents(latents: mx.array) -> mx.array:
        batch_size, num_channels, height, width = latents.shape
        return latents.reshape(batch_size, num_channels, height * width).transpose(0, 2, 1)

    @staticmethod
    def prepare_latent_ids(latents: mx.array) -> mx.array:
        batch_size, _, height, width = latents.shape
        h_ids = mx.arange(height, dtype=mx.int32)
        w_ids = mx.arange(width, dtype=mx.int32)
        h_grid = mx.broadcast_to(mx.expand_dims(h_ids, axis=1), (height, width))
        w_grid = mx.broadcast_to(mx.expand_dims(w_ids, axis=0), (height, width))
        flat_h = h_grid.reshape(-1)
        flat_w = w_grid.reshape(-1)
        t = mx.zeros_like(flat_h)
        layer_ids = mx.zeros_like(flat_h)
        coords = mx.stack([t, flat_h, flat_w, layer_ids], axis=1)
        coords = mx.expand_dims(coords, axis=0)
        return mx.broadcast_to(coords, (batch_size, coords.shape[1], coords.shape[2]))

    @staticmethod
    def prepare_latents(
        seed: int,
        height: int,
        width: int,
        batch_size: int,
        num_latents_channels: int = 32,
        vae_scale_factor: int = 8,
    ) -> tuple[mx.array, mx.array, int, int]:
        height = 2 * (height // (vae_scale_factor * 2))
        width = 2 * (width // (vae_scale_factor * 2))
        latent_height = height // 2
        latent_width = width // 2
        latents = mx.random.normal(
            shape=(batch_size, num_latents_channels * 4, latent_height, latent_width),
            key=mx.random.key(seed),
        ).astype(ModelConfig.precision)
        latent_ids = Flux2LatentCreator.prepare_latent_ids(latents)
        return latents, latent_ids, latent_height, latent_width

    @staticmethod
    def prepare_latent_ids_from_packed(latents: mx.array) -> mx.array:
        batch_size, seq_len, _ = latents.shape
        height = int(mx.sqrt(mx.array(seq_len, dtype=mx.float32)).item())
        width = seq_len // height if height > 0 else 0
        h_ids = mx.arange(height, dtype=mx.int32)
        w_ids = mx.arange(width, dtype=mx.int32)
        h_grid = mx.broadcast_to(mx.expand_dims(h_ids, axis=1), (height, width))
        w_grid = mx.broadcast_to(mx.expand_dims(w_ids, axis=0), (height, width))
        flat_h = h_grid.reshape(-1)
        flat_w = w_grid.reshape(-1)
        t = mx.zeros_like(flat_h)
        layer_ids = mx.zeros_like(flat_h)
        coords = mx.stack([t, flat_h, flat_w, layer_ids], axis=1)
        coords = mx.expand_dims(coords, axis=0)
        return mx.broadcast_to(coords, (batch_size, coords.shape[1], coords.shape[2]))
