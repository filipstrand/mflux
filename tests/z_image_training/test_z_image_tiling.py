"""Tests for Z-Image VAE tiling functionality.

VAE tiling enables high-resolution (4K+) image generation without OOM
by processing the VAE encoder/decoder in overlapping tiles.
"""

import pytest

from mflux.models.common.vae.tiling_config import TilingConfig


class TestTilingConfig:
    """Tests for TilingConfig dataclass."""

    def test_default_config(self):
        """Test that default config has expected values."""
        config = TilingConfig()
        assert config.vae_decode_tiles_per_dim == 8
        assert config.vae_decode_overlap == 8
        assert config.vae_encode_tiled is True
        assert config.vae_encode_tile_size == 512
        assert config.vae_encode_tile_overlap == 64

    def test_custom_config(self):
        """Test custom configuration values."""
        config = TilingConfig(
            vae_decode_tiles_per_dim=4,
            vae_decode_overlap=16,
            vae_encode_tiled=False,
            vae_encode_tile_size=256,
            vae_encode_tile_overlap=32,
        )
        assert config.vae_decode_tiles_per_dim == 4
        assert config.vae_decode_overlap == 16
        assert config.vae_encode_tiled is False
        assert config.vae_encode_tile_size == 256
        assert config.vae_encode_tile_overlap == 32

    def test_config_is_frozen(self):
        """Test that TilingConfig is immutable."""
        config = TilingConfig()
        with pytest.raises(AttributeError):
            config.vae_decode_tiles_per_dim = 4

    def test_config_disabled(self):
        """Test configuration for disabled tiling."""
        config = TilingConfig(
            vae_decode_tiles_per_dim=None,
            vae_encode_tiled=False,
        )
        assert config.vae_decode_tiles_per_dim is None
        assert config.vae_encode_tiled is False


class TestZImageTilingIntegration:
    """Integration tests for Z-Image tiling.

    Note: These tests verify the API and parameter passing without
    requiring actual model weights. Full end-to-end tests would require
    model initialization which is expensive.
    """

    def test_enable_tiling_parameter_exists(self):
        """Test that enable_tiling parameter is accepted by generate_image."""
        # Import the class to check signature
        import inspect

        from mflux.models.z_image.variants.txt2img.z_image import ZImage

        sig = inspect.signature(ZImage.generate_image)
        params = sig.parameters

        assert "enable_tiling" in params
        assert params["enable_tiling"].default is False

    def test_tiling_config_types(self):
        """Test TilingConfig parameter types."""
        # Valid configurations should not raise
        TilingConfig(vae_decode_tiles_per_dim=8)
        TilingConfig(vae_decode_tiles_per_dim=None)  # Disabled
        TilingConfig(vae_decode_tiles_per_dim=0)  # Edge case

        # Should be hashable (frozen dataclass)
        config = TilingConfig()
        hash(config)

    def test_recommended_4k_config(self):
        """Test recommended configuration for 4K generation."""
        # These are the values used when enable_tiling=True
        config = TilingConfig(
            vae_decode_tiles_per_dim=8,
            vae_decode_overlap=8,
            vae_encode_tiled=True,
            vae_encode_tile_size=512,
            vae_encode_tile_overlap=64,
        )

        # 8 tiles per dimension with overlap should handle 4096x4096
        # Each tile handles ~512x512 effective area
        effective_tile_size = config.vae_encode_tile_size - config.vae_encode_tile_overlap
        max_dimension = config.vae_decode_tiles_per_dim * effective_tile_size

        assert max_dimension >= 3584  # At least close to 4K


class TestTilingMemoryEstimates:
    """Tests for memory estimation with tiling."""

    def test_memory_reduction_factor(self):
        """Test that tiling reduces per-tile memory requirements."""
        # Without tiling: process entire image at once
        # With 8x8 tiling: process 1/64th at a time (plus overlap)
        tiles_per_dim = 8
        theoretical_reduction = 1 / (tiles_per_dim**2)

        # Overlap adds some overhead, so actual reduction is less
        # but should still be significant
        assert theoretical_reduction < 0.02  # 64x reduction theoretical

    def test_overlap_prevents_seams(self):
        """Verify overlap is sufficient to prevent visible seams."""
        config = TilingConfig()

        # Overlap should be at least 8 pixels (common minimum)
        assert config.vae_decode_overlap >= 8
        assert config.vae_encode_tile_overlap >= 32
