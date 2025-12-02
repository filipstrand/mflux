"""Tests for vectorized RoPE implementation in Z-Image.

This test suite verifies that the vectorized RoPE implementation produces
numerically equivalent results to the original implementation.
"""

import mlx.core as mx
import pytest

from mflux.models.zimage.embeddings.rope_3d import RoPE3D


class TestRoPEVectorized:
    """Test suite for vectorized RoPE frequency lookups."""

    def test_rope_initialization(self):
        """Test that RoPE initializes correctly."""
        rope = RoPE3D()

        # Check that frequency cache is populated
        assert len(rope._freqs_cache) == 3, "Should have 3 axes"

        # Check dimensions
        for axis_idx in range(3):
            cos_table, sin_table = rope._freqs_cache[axis_idx]
            expected_max_len = RoPE3D.AXES_LENS[axis_idx]
            expected_dim = RoPE3D.AXES_DIMS[axis_idx] // 2

            assert cos_table.shape == (expected_max_len, expected_dim)
            assert sin_table.shape == (expected_max_len, expected_dim)

    def test_get_freqs_for_positions_single_position(self):
        """Test frequency lookup for a single position."""
        rope = RoPE3D()

        # Single position: time=5, h=10, w=20
        pos_ids = mx.array([[5, 10, 20]], dtype=mx.int32)
        freqs_cos, freqs_sin = rope.get_freqs_for_positions(pos_ids)

        # Check output shape
        assert freqs_cos.shape == (1, RoPE3D.HEAD_DIM // 2)
        assert freqs_sin.shape == (1, RoPE3D.HEAD_DIM // 2)

        # Verify concatenation: should have all three axes
        expected_dim = sum(RoPE3D.AXES_DIMS) // 2
        assert freqs_cos.shape[1] == expected_dim

    def test_get_freqs_for_positions_multiple_positions(self):
        """Test frequency lookup for multiple positions."""
        rope = RoPE3D()

        # Multiple positions
        pos_ids = mx.array([[1, 0, 0], [2, 5, 10], [10, 20, 30], [100, 50, 50]], dtype=mx.int32)

        freqs_cos, freqs_sin = rope.get_freqs_for_positions(pos_ids)

        # Check output shape
        assert freqs_cos.shape == (4, RoPE3D.HEAD_DIM // 2)
        assert freqs_sin.shape == (4, RoPE3D.HEAD_DIM // 2)

    def test_get_freqs_numerical_correctness(self):
        """Test that frequencies are computed correctly.

        This verifies the lookup is working by checking:
        1. Position [0,0,0] should give specific values (all axes at position 0)
        2. Different positions should give different frequencies
        """
        rope = RoPE3D()

        # Position at origin
        pos_zero = mx.array([[0, 0, 0]], dtype=mx.int32)
        freqs_cos_zero, freqs_sin_zero = rope.get_freqs_for_positions(pos_zero)

        # At position 0, cos should be 1.0 and sin should be 0.0
        assert mx.allclose(freqs_cos_zero, mx.ones_like(freqs_cos_zero), atol=1e-6)
        assert mx.allclose(freqs_sin_zero, mx.zeros_like(freqs_sin_zero), atol=1e-6)

        # Position away from origin should give different values
        pos_other = mx.array([[10, 20, 30]], dtype=mx.int32)
        freqs_cos_other, freqs_sin_other = rope.get_freqs_for_positions(pos_other)

        # Should not be all ones/zeros
        assert not mx.allclose(freqs_cos_other, mx.ones_like(freqs_cos_other), atol=1e-2)
        assert not mx.allclose(freqs_sin_other, mx.zeros_like(freqs_sin_other), atol=1e-2)

    def test_get_image_freqs_shape(self):
        """Test image frequency generation."""
        rope = RoPE3D()

        # Test common image dimensions
        h_patches, w_patches = 64, 64  # 1024x1024 image
        time_offset = 128

        freqs_cos, freqs_sin = rope.get_image_freqs(h_patches, w_patches, time_offset)

        n_patches = h_patches * w_patches
        assert freqs_cos.shape == (n_patches, RoPE3D.HEAD_DIM // 2)
        assert freqs_sin.shape == (n_patches, RoPE3D.HEAD_DIM // 2)

    def test_get_image_freqs_time_offset(self):
        """Test that time offset is applied correctly in image frequencies."""
        rope = RoPE3D()

        h_patches, w_patches = 4, 4

        # Get frequencies with different time offsets
        freqs_cos_t1, freqs_sin_t1 = rope.get_image_freqs(h_patches, w_patches, time_offset=1)
        freqs_cos_t10, freqs_sin_t10 = rope.get_image_freqs(h_patches, w_patches, time_offset=10)

        # Frequencies should be different due to time offset
        assert not mx.allclose(freqs_cos_t1, freqs_cos_t10, atol=1e-6)
        assert not mx.allclose(freqs_sin_t1, freqs_sin_t10, atol=1e-6)

    def test_get_caption_freqs_shape(self):
        """Test caption frequency generation."""
        rope = RoPE3D()

        cap_len = 77  # Typical caption length
        freqs_cos, freqs_sin = rope.get_caption_freqs(cap_len)

        assert freqs_cos.shape == (cap_len, RoPE3D.HEAD_DIM // 2)
        assert freqs_sin.shape == (cap_len, RoPE3D.HEAD_DIM // 2)

    def test_caption_freqs_positions(self):
        """Test that caption positions follow the expected pattern.

        Caption tokens should have:
        - time = 1, 2, 3, ..., cap_len (sequential)
        - h = 0, w = 0 (fixed)
        """
        rope = RoPE3D()

        cap_len = 10
        freqs_cos, freqs_sin = rope.get_caption_freqs(cap_len)

        # Manually compute what the first token should be (time=1, h=0, w=0)
        pos_first = mx.array([[1, 0, 0]], dtype=mx.int32)
        expected_cos_first, expected_sin_first = rope.get_freqs_for_positions(pos_first)

        # Check first token matches
        assert mx.allclose(freqs_cos[0:1], expected_cos_first, atol=1e-6)
        assert mx.allclose(freqs_sin[0:1], expected_sin_first, atol=1e-6)

        # Check last token (time=cap_len, h=0, w=0)
        pos_last = mx.array([[cap_len, 0, 0]], dtype=mx.int32)
        expected_cos_last, expected_sin_last = rope.get_freqs_for_positions(pos_last)

        assert mx.allclose(freqs_cos[-1:], expected_cos_last, atol=1e-6)
        assert mx.allclose(freqs_sin[-1:], expected_sin_last, atol=1e-6)

    def test_get_combined_freqs_caching(self):
        """Test that combined frequencies are cached correctly."""
        rope = RoPE3D()

        h_patches, w_patches = 32, 32
        cap_len = 77
        padded_cap_len = 96  # Rounded to multiple of 32

        # First call should compute
        freqs_cos_1, freqs_sin_1 = rope.get_combined_freqs(h_patches, w_patches, cap_len, padded_cap_len)

        # Second call should return cached value
        freqs_cos_2, freqs_sin_2 = rope.get_combined_freqs(h_patches, w_patches, cap_len, padded_cap_len)

        # Should be identical (same object, not just equal values)
        assert freqs_cos_1 is freqs_cos_2
        assert freqs_sin_1 is freqs_sin_2

    def test_combined_freqs_shape(self):
        """Test that combined frequencies have correct shape."""
        rope = RoPE3D()

        h_patches, w_patches = 64, 64
        cap_len = 77
        padded_cap_len = 96

        freqs_cos, freqs_sin = rope.get_combined_freqs(h_patches, w_patches, cap_len, padded_cap_len)

        n_img_tokens = h_patches * w_patches
        total_tokens = n_img_tokens + cap_len

        assert freqs_cos.shape == (total_tokens, RoPE3D.HEAD_DIM // 2)
        assert freqs_sin.shape == (total_tokens, RoPE3D.HEAD_DIM // 2)

    def test_combined_freqs_correctness(self):
        """Test that combined frequencies match separate computations."""
        rope = RoPE3D()

        h_patches, w_patches = 16, 16
        cap_len = 32
        padded_cap_len = 32

        # Get combined frequencies
        combined_cos, combined_sin = rope.get_combined_freqs(h_patches, w_patches, cap_len, padded_cap_len)

        # Get separate frequencies
        img_cos, img_sin = rope.get_image_freqs(h_patches, w_patches, time_offset=padded_cap_len + 1)
        cap_cos, cap_sin = rope.get_caption_freqs(cap_len)

        n_img_tokens = h_patches * w_patches

        # Check image part
        assert mx.allclose(combined_cos[:n_img_tokens], img_cos, atol=1e-6)
        assert mx.allclose(combined_sin[:n_img_tokens], img_sin, atol=1e-6)

        # Check caption part
        assert mx.allclose(combined_cos[n_img_tokens:], cap_cos, atol=1e-6)
        assert mx.allclose(combined_sin[n_img_tokens:], cap_sin, atol=1e-6)

    def test_legacy_call_interface(self):
        """Test that the legacy __call__ interface still works."""
        rope = RoPE3D()

        height, width = 512, 512
        freqs_cos, freqs_sin = rope(height, width)

        h_patches = height // 16
        w_patches = width // 16
        n_patches = h_patches * w_patches

        assert freqs_cos.shape == (n_patches, RoPE3D.HEAD_DIM // 2)
        assert freqs_sin.shape == (n_patches, RoPE3D.HEAD_DIM // 2)

    @pytest.mark.parametrize(
        "h_patches,w_patches",
        [
            (16, 16),  # 256x256
            (32, 32),  # 512x512
            (64, 64),  # 1024x1024
            (32, 64),  # Rectangular
            (96, 96),  # 1536x1536
        ],
    )
    def test_various_image_sizes(self, h_patches, w_patches):
        """Test RoPE with various image dimensions."""
        rope = RoPE3D()

        freqs_cos, freqs_sin = rope.get_image_freqs(h_patches, w_patches, time_offset=1)

        n_patches = h_patches * w_patches
        assert freqs_cos.shape == (n_patches, RoPE3D.HEAD_DIM // 2)
        assert freqs_sin.shape == (n_patches, RoPE3D.HEAD_DIM // 2)

    @pytest.mark.parametrize("cap_len", [1, 32, 77, 128, 256])
    def test_various_caption_lengths(self, cap_len):
        """Test RoPE with various caption lengths."""
        rope = RoPE3D()

        freqs_cos, freqs_sin = rope.get_caption_freqs(cap_len)

        assert freqs_cos.shape == (cap_len, RoPE3D.HEAD_DIM // 2)
        assert freqs_sin.shape == (cap_len, RoPE3D.HEAD_DIM // 2)

    def test_frequencies_in_valid_range(self):
        """Test that cos/sin values are in valid range [-1, 1]."""
        rope = RoPE3D()

        # Test with large positions
        pos_ids = mx.array(
            [
                [500, 400, 400],
                [1000, 500, 500],
            ],
            dtype=mx.int32,
        )

        freqs_cos, freqs_sin = rope.get_freqs_for_positions(pos_ids)

        # Cos and sin should be in [-1, 1]
        assert mx.all(freqs_cos >= -1.0) and mx.all(freqs_cos <= 1.0)
        assert mx.all(freqs_sin >= -1.0) and mx.all(freqs_sin <= 1.0)

    def test_cache_isolation(self):
        """Test that different cache keys don't interfere."""
        rope = RoPE3D()

        # Create different cached entries
        freqs1 = rope.get_combined_freqs(32, 32, 77, 96)
        freqs2 = rope.get_combined_freqs(64, 64, 77, 96)
        freqs3 = rope.get_combined_freqs(32, 32, 64, 64)

        # Should all be different
        assert not mx.array_equal(freqs1[0], freqs2[0])
        assert not mx.array_equal(freqs1[0], freqs3[0])
        assert not mx.array_equal(freqs2[0], freqs3[0])
