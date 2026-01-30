"""Tests for Z-Image batch inference utilities.

Tests batched latent creation and manipulation.
"""

import mlx.core as mx
import numpy as np
import pytest

from mflux.models.z_image.latent_creator import ZImageLatentCreator


@pytest.mark.fast
def test_create_noise_batch_shape():
    """Test that batched noise has correct shape."""
    seeds = [1, 2, 3, 4]
    height = 64
    width = 64

    result = ZImageLatentCreator.create_noise_batch(
        seeds=seeds,
        height=height,
        width=width,
    )

    # Should be [batch_size, 16, 1, H/8, W/8]
    expected_shape = (4, 16, 1, 8, 8)
    assert result.shape == expected_shape


@pytest.mark.fast
def test_create_noise_batch_single():
    """Test batch with single seed."""
    seeds = [42]
    height = 64
    width = 64

    result = ZImageLatentCreator.create_noise_batch(
        seeds=seeds,
        height=height,
        width=width,
    )

    assert result.shape == (1, 16, 1, 8, 8)


@pytest.mark.fast
def test_create_noise_batch_reproducible():
    """Test that same seeds produce same results."""
    seeds = [1, 2, 3]
    height = 64
    width = 64

    result1 = ZImageLatentCreator.create_noise_batch(
        seeds=seeds,
        height=height,
        width=width,
    )

    result2 = ZImageLatentCreator.create_noise_batch(
        seeds=seeds,
        height=height,
        width=width,
    )

    # Convert to float32 for numpy compatibility (bfloat16 has issues)
    np.testing.assert_array_equal(
        np.array(result1.astype(mx.float32)),
        np.array(result2.astype(mx.float32)),
    )


@pytest.mark.fast
def test_create_noise_batch_different_seeds():
    """Test that different seeds produce different results."""
    height = 64
    width = 64

    result1 = ZImageLatentCreator.create_noise_batch(
        seeds=[1],
        height=height,
        width=width,
    )

    result2 = ZImageLatentCreator.create_noise_batch(
        seeds=[2],
        height=height,
        width=width,
    )

    # Convert to float32 for numpy compatibility (bfloat16 has issues)
    # Should not be equal
    assert not np.allclose(
        np.array(result1.astype(mx.float32)),
        np.array(result2.astype(mx.float32)),
    )


@pytest.mark.fast
def test_create_noise_batch_matches_single():
    """Test that batched noise matches single noise for same seed."""
    seed = 42
    height = 64
    width = 64

    # Create single noise
    single_noise = ZImageLatentCreator.create_noise(
        seed=seed,
        height=height,
        width=width,
    )

    # Create batched noise with same seed
    batch_noise = ZImageLatentCreator.create_noise_batch(
        seeds=[seed],
        height=height,
        width=width,
    )

    # Extract first batch item and compare
    batch_first = batch_noise[0]

    # Convert to float32 for numpy compatibility (bfloat16 has issues)
    np.testing.assert_array_almost_equal(
        np.array(single_noise.astype(mx.float32)),
        np.array(batch_first.astype(mx.float32)),
        decimal=5,
    )


@pytest.mark.fast
def test_create_noise_batch_larger_size():
    """Test batch with larger image size."""
    seeds = [1, 2]
    height = 512
    width = 512

    result = ZImageLatentCreator.create_noise_batch(
        seeds=seeds,
        height=height,
        width=width,
    )

    # 512/8 = 64
    expected_shape = (2, 16, 1, 64, 64)
    assert result.shape == expected_shape


@pytest.mark.fast
def test_create_noise_batch_non_square():
    """Test batch with non-square dimensions."""
    seeds = [1, 2, 3]
    height = 128
    width = 64

    result = ZImageLatentCreator.create_noise_batch(
        seeds=seeds,
        height=height,
        width=width,
    )

    # 128/8 = 16, 64/8 = 8
    expected_shape = (3, 16, 1, 16, 8)
    assert result.shape == expected_shape


@pytest.mark.fast
def test_pack_latents_batch_5d():
    """Test packing 5D latents (already packed)."""
    # Already packed [B, C, 1, H, W]
    latents = mx.ones((2, 16, 1, 8, 8))

    result = ZImageLatentCreator.pack_latents_batch(
        latents=latents,
        height=64,
        width=64,
    )

    # Should return unchanged
    assert result.shape == (2, 16, 1, 8, 8)


@pytest.mark.fast
def test_pack_latents_batch_4d():
    """Test packing 4D latents (needs frame dimension)."""
    # Unpacked [B, C, H, W]
    latents = mx.ones((2, 16, 8, 8))

    result = ZImageLatentCreator.pack_latents_batch(
        latents=latents,
        height=64,
        width=64,
    )

    # Should add frame dimension
    assert result.shape == (2, 16, 1, 8, 8)


@pytest.mark.fast
def test_unpack_latents_batch_5d():
    """Test unpacking 5D latents."""
    # Packed [B, C, 1, H, W]
    latents = mx.ones((2, 16, 1, 8, 8))

    result = ZImageLatentCreator.unpack_latents_batch(
        latents=latents,
        height=64,
        width=64,
    )

    # Should remove frame dimension
    assert result.shape == (2, 16, 8, 8)


@pytest.mark.fast
def test_unpack_latents_batch_4d():
    """Test unpacking 4D latents (already unpacked)."""
    # Already unpacked [B, C, H, W]
    latents = mx.ones((2, 16, 8, 8))

    result = ZImageLatentCreator.unpack_latents_batch(
        latents=latents,
        height=64,
        width=64,
    )

    # Should return unchanged
    assert result.shape == (2, 16, 8, 8)


@pytest.mark.fast
def test_pack_unpack_roundtrip():
    """Test that pack/unpack is reversible."""
    # Start with 4D latents
    original = mx.random.normal(shape=(4, 16, 8, 8), key=mx.random.key(42))

    # Pack
    packed = ZImageLatentCreator.pack_latents_batch(original, 64, 64)
    assert packed.shape == (4, 16, 1, 8, 8)

    # Unpack
    unpacked = ZImageLatentCreator.unpack_latents_batch(packed, 64, 64)
    assert unpacked.shape == (4, 16, 8, 8)

    # Should match original
    np.testing.assert_array_almost_equal(
        np.array(original),
        np.array(unpacked),
        decimal=5,
    )


@pytest.mark.fast
def test_create_noise_statistics():
    """Test that generated noise has expected statistical properties."""
    # Use max batch size (64) to test statistics
    seeds = list(range(64))
    height = 64
    width = 64

    result = ZImageLatentCreator.create_noise_batch(
        seeds=seeds,
        height=height,
        width=width,
    )

    # Convert to float32 for numpy compatibility (bfloat16 has issues)
    result_np = np.array(result.astype(mx.float32))

    # Mean should be close to 0
    assert abs(result_np.mean()) < 0.1

    # Std should be close to 1
    assert abs(result_np.std() - 1.0) < 0.1


@pytest.mark.fast
def test_create_noise_batch_empty_seeds_raises():
    """Test that empty seeds list raises ValueError."""
    with pytest.raises(ValueError, match="cannot be empty"):
        ZImageLatentCreator.create_noise_batch(
            seeds=[],
            height=64,
            width=64,
        )


@pytest.mark.fast
def test_create_noise_batch_too_many_seeds_raises():
    """Test that exceeding max batch size raises ValueError."""
    with pytest.raises(ValueError, match="too large"):
        ZImageLatentCreator.create_noise_batch(
            seeds=list(range(100)),  # 100 > MAX_BATCH_SIZE (64)
            height=64,
            width=64,
        )


@pytest.mark.fast
def test_create_noise_batch_invalid_height_raises():
    """Test that out-of-bounds height raises ValueError."""
    # Height too small
    with pytest.raises(ValueError, match="height must be between"):
        ZImageLatentCreator.create_noise_batch(
            seeds=[1],
            height=32,  # Below MIN_DIMENSION (64)
            width=64,
        )

    # Height too large
    with pytest.raises(ValueError, match="height must be between"):
        ZImageLatentCreator.create_noise_batch(
            seeds=[1],
            height=8192,  # Above MAX_DIMENSION (4096)
            width=64,
        )


@pytest.mark.fast
def test_create_noise_batch_invalid_width_raises():
    """Test that out-of-bounds width raises ValueError."""
    # Width too small
    with pytest.raises(ValueError, match="width must be between"):
        ZImageLatentCreator.create_noise_batch(
            seeds=[1],
            height=64,
            width=32,  # Below MIN_DIMENSION (64)
        )

    # Width too large
    with pytest.raises(ValueError, match="width must be between"):
        ZImageLatentCreator.create_noise_batch(
            seeds=[1],
            height=64,
            width=8192,  # Above MAX_DIMENSION (4096)
        )


@pytest.mark.fast
def test_create_noise_batch_non_list_raises():
    """Test that non-list seeds raises TypeError."""
    with pytest.raises(TypeError, match="must be a list"):
        ZImageLatentCreator.create_noise_batch(
            seeds=42,  # Not a list
            height=64,
            width=64,
        )


@pytest.mark.fast
def test_create_noise_batch_non_integer_seed_raises():
    """Test that non-integer seeds in list raise TypeError."""
    with pytest.raises(TypeError, match="seed at index 1 must be an integer"):
        ZImageLatentCreator.create_noise_batch(
            seeds=[1, "bad", 3],  # String in list
            height=64,
            width=64,
        )

    with pytest.raises(TypeError, match="seed at index 0 must be an integer"):
        ZImageLatentCreator.create_noise_batch(
            seeds=[None, 2, 3],  # None in list
            height=64,
            width=64,
        )


@pytest.mark.fast
def test_create_noise_single_validation():
    """Test that single noise creation also validates dimensions."""
    # Height too small
    with pytest.raises(ValueError, match="height must be between"):
        ZImageLatentCreator.create_noise(
            seed=1,
            height=32,
            width=64,
        )

    # Width too large
    with pytest.raises(ValueError, match="width must be between"):
        ZImageLatentCreator.create_noise(
            seed=1,
            height=64,
            width=8192,
        )
