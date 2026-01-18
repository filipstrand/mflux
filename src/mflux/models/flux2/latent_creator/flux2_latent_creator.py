import mlx.core as mx

from mflux.models.common.config.model_config import ModelConfig


class Flux2LatentCreator:
    @staticmethod
    def patchify_latents(latents: mx.array) -> mx.array:
        if latents.ndim == 5 and latents.shape[2] == 1:
            latents = latents[:, :, 0, :, :]
        if latents.ndim != 4:
            raise ValueError(f"Expected latents with ndim=4, got shape={latents.shape}")
        batch_size, num_channels, height, width = latents.shape
        latents = latents.reshape(batch_size, num_channels, height // 2, 2, width // 2, 2)
        latents = latents.transpose(0, 1, 3, 5, 2, 4)
        latents = latents.reshape(batch_size, num_channels * 4, height // 2, width // 2)
        return latents

    @staticmethod
    def pack_latents(latents: mx.array) -> mx.array:
        batch_size, num_channels, height, width = latents.shape
        return latents.reshape(batch_size, num_channels, height * width).transpose(0, 2, 1)

    @staticmethod
    def unpack_latents(latents: mx.array, height: int, width: int, vae_scale_factor: int = 8) -> mx.array:
        """
        Convert packed Flux2 latents (B, seq, C) into 4D packed-latent layout (B, C, H, W)
        where H=W=floor(image_dim / (vae_scale_factor * 2)).
        """
        if latents.ndim == 4:
            return latents

        if latents.ndim != 3:
            raise ValueError(f"Expected packed latents with ndim=3, got shape={latents.shape}")

        batch_size, seq_len, channels = latents.shape
        latent_height = height // (vae_scale_factor * 2)
        latent_width = width // (vae_scale_factor * 2)
        expected_seq_len = latent_height * latent_width
        if expected_seq_len != seq_len:
            raise ValueError(
                f"Packed latent seq_len mismatch: got {seq_len}, expected {expected_seq_len} "
                f"for height={height} width={width} (latent {latent_height}x{latent_width})"
            )

        return latents.reshape(batch_size, latent_height, latent_width, channels).transpose(0, 3, 1, 2)

    @staticmethod
    def prepare_grid_ids(latents: mx.array, *, t_coord: int) -> mx.array:
        batch_size, _, height, width = latents.shape
        h_ids = mx.arange(height, dtype=mx.int32)
        w_ids = mx.arange(width, dtype=mx.int32)
        h_grid = mx.broadcast_to(mx.expand_dims(h_ids, axis=1), (height, width))
        w_grid = mx.broadcast_to(mx.expand_dims(w_ids, axis=0), (height, width))
        flat_h = h_grid.reshape(-1)
        flat_w = w_grid.reshape(-1)
        t = mx.full(flat_h.shape, t_coord, dtype=mx.int32)
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
        latent_ids = Flux2LatentCreator.prepare_grid_ids(latents, t_coord=0)
        return latents, latent_ids, latent_height, latent_width

    @staticmethod
    def prepare_packed_latents(
        seed: int,
        height: int,
        width: int,
        batch_size: int,
        num_latents_channels: int = 32,
        vae_scale_factor: int = 8,
    ) -> tuple[mx.array, mx.array, int, int]:
        latents, latent_ids, latent_height, latent_width = Flux2LatentCreator.prepare_latents(
            seed=seed,
            height=height,
            width=width,
            batch_size=batch_size,
            num_latents_channels=num_latents_channels,
            vae_scale_factor=vae_scale_factor,
        )
        return Flux2LatentCreator.pack_latents(latents), latent_ids, latent_height, latent_width
