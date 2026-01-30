import mlx.core as mx

from mflux.models.common.config import ModelConfig
from mflux.models.z_image.constants import MAX_BATCH_SIZE, MAX_DIMENSION, MIN_DIMENSION


class ZImageLatentCreator:
    # Re-export constants for backwards compatibility
    MAX_BATCH_SIZE = MAX_BATCH_SIZE
    MAX_DIMENSION = MAX_DIMENSION
    MIN_DIMENSION = MIN_DIMENSION

    @staticmethod
    def _validate_dimensions(height: int, width: int) -> None:
        """Validate image dimensions are within safe bounds.

        Raises:
            ValueError: If dimensions are outside allowed range.
        """
        if not isinstance(height, int) or not isinstance(width, int):
            raise TypeError("height and width must be integers")
        if height < ZImageLatentCreator.MIN_DIMENSION or height > ZImageLatentCreator.MAX_DIMENSION:
            raise ValueError(
                f"height must be between {ZImageLatentCreator.MIN_DIMENSION} and "
                f"{ZImageLatentCreator.MAX_DIMENSION}, got {height}"
            )
        if width < ZImageLatentCreator.MIN_DIMENSION or width > ZImageLatentCreator.MAX_DIMENSION:
            raise ValueError(
                f"width must be between {ZImageLatentCreator.MIN_DIMENSION} and "
                f"{ZImageLatentCreator.MAX_DIMENSION}, got {width}"
            )

    @staticmethod
    def create_noise(seed: int, height: int, width: int) -> mx.array:
        ZImageLatentCreator._validate_dimensions(height, width)
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
    def create_noise_batch(seeds: list[int], height: int, width: int) -> mx.array:
        """Create batched noise latents for parallel generation.

        Args:
            seeds: List of random seeds (one per batch item, max 64)
            height: Image height (64-4096)
            width: Image width (64-4096)

        Returns:
            Batched noise array [batch_size, 16, 1, H/8, W/8]

        Raises:
            ValueError: If seeds list is empty or exceeds MAX_BATCH_SIZE,
                       or if dimensions are out of bounds.
            TypeError: If seeds is not a list or dimensions are not integers.
        """
        # Validate inputs
        if not isinstance(seeds, list):
            raise TypeError("seeds must be a list of integers")
        if len(seeds) == 0:
            raise ValueError("seeds list cannot be empty")
        if len(seeds) > ZImageLatentCreator.MAX_BATCH_SIZE:
            raise ValueError(
                f"seeds list too large: {len(seeds)} exceeds maximum of {ZImageLatentCreator.MAX_BATCH_SIZE}"
            )
        # Validate each seed is an integer
        for i, seed in enumerate(seeds):
            if not isinstance(seed, int):
                raise TypeError(f"seed at index {i} must be an integer, got {type(seed).__name__}")
        ZImageLatentCreator._validate_dimensions(height, width)

        latent_h = height // 8
        latent_w = width // 8

        # Create noise arrays and stack. Memory is bounded by MAX_BATCH_SIZE (64).
        # At 1024x1024 with bfloat16: 64 * 16 * 1 * 128 * 128 * 2 bytes = ~32MB max.
        noise_arrays = [
            mx.random.normal(
                shape=[16, 1, latent_h, latent_w],
                key=mx.random.key(seed),
            ).astype(ModelConfig.precision)
            for seed in seeds
        ]

        # Stack into batch
        return mx.stack(noise_arrays, axis=0)  # [B, 16, 1, H/8, W/8]

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

    @staticmethod
    def pack_latents_batch(latents: mx.array, height: int, width: int) -> mx.array:  # noqa: ARG004
        """Pack batched latents for transformer input.

        Args:
            latents: Batched latents [B, C, 1, H, W] or [B, C, H, W]
            height: Image height (unused, for API compatibility)
            width: Image width (unused, for API compatibility)

        Returns:
            Packed latents [B, C, 1, H, W]
        """
        if latents.ndim == 5:
            return latents
        # Add frame dimension if missing
        return mx.expand_dims(latents, axis=2)

    @staticmethod
    def unpack_latents_batch(latents: mx.array, height: int, width: int) -> mx.array:  # noqa: ARG004
        """Unpack batched latents from transformer output.

        Args:
            latents: Batched latents [B, C, 1, H, W]
            height: Image height (unused, for API compatibility)
            width: Image width (unused, for API compatibility)

        Returns:
            Unpacked latents [B, C, H, W]
        """
        if latents.ndim == 5:
            return mx.squeeze(latents, axis=2)
        return latents
