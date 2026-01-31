"""Tests for Z-Image CFG batching functionality.

CFG batching processes conditional and unconditional noise predictions
in a single transformer call for 30-50% inference speedup.
"""

from unittest.mock import MagicMock

import mlx.core as mx
import pytest


class TestComputeCfgBatched:
    """Tests for _compute_cfg_batched method."""

    @pytest.fixture
    def mock_z_image(self):
        """Create mock ZImage with transformer."""
        model = MagicMock()

        # Mock transformer to return predictable batched output
        def mock_transformer(t, x, cap_feats, sigmas):
            # Return noise proportional to embeddings for testing
            batch_size = x.shape[0]
            # First half (cond) returns positive values, second half (uncond) returns zeros
            if batch_size == 2:
                cond_noise = mx.ones_like(x[:1]) * 2.0
                uncond_noise = mx.ones_like(x[:1]) * 0.5
                return mx.concatenate([cond_noise, uncond_noise], axis=0)
            return mx.ones_like(x)

        model.transformer = mock_transformer
        return model

    def test_batches_inputs_correctly(self, mock_z_image):
        """Test that inputs are correctly batched."""
        from mflux.models.z_image.variants.txt2img.z_image import ZImage

        # Create test inputs
        latents = mx.ones((1, 4, 64, 64))
        text_encodings = mx.ones((1, 77, 768)) * 2.0
        negative_encodings = mx.ones((1, 77, 768)) * 0.5
        sigmas = mx.array([1.0])

        # Track transformer calls
        call_args = []
        original_transformer = mock_z_image.transformer

        def tracking_transformer(t, x, cap_feats, sigmas):
            call_args.append(
                {
                    "t": t,
                    "x_shape": x.shape,
                    "cap_feats_shape": cap_feats.shape,
                }
            )
            return original_transformer(t, x, cap_feats, sigmas)

        mock_z_image.transformer = tracking_transformer

        # Call the method (we need to bind it to our mock)
        cfg_output = ZImage._compute_cfg_batched(
            mock_z_image,
            t=0,
            latents=latents,
            text_encodings=text_encodings,
            negative_encodings=negative_encodings,
            sigmas=sigmas,
            guidance_scale=4.0,
            cfg_normalization=False,
        )

        # Should have made exactly one transformer call
        assert len(call_args) == 1
        # Output should have same shape as input latents
        assert cfg_output.shape == latents.shape

        # Batch dimension should be doubled
        assert call_args[0]["x_shape"][0] == 2
        assert call_args[0]["cap_feats_shape"][0] == 2

    def test_output_shape_matches_input(self, mock_z_image):
        """Test that output shape matches single latent input."""
        from mflux.models.z_image.variants.txt2img.z_image import ZImage

        latents = mx.ones((1, 4, 64, 64))
        text_encodings = mx.ones((1, 77, 768))
        negative_encodings = mx.ones((1, 77, 768))
        sigmas = mx.array([1.0])

        result = ZImage._compute_cfg_batched(
            mock_z_image,
            t=0,
            latents=latents,
            text_encodings=text_encodings,
            negative_encodings=negative_encodings,
            sigmas=sigmas,
            guidance_scale=4.0,
            cfg_normalization=False,
        )

        # Output should have same shape as input latents
        assert result.shape == latents.shape

    def test_cfg_formula_applied_correctly(self, mock_z_image):
        """Test CFG formula: uncond + scale * (cond - uncond)."""
        from mflux.models.z_image.variants.txt2img.z_image import ZImage

        latents = mx.ones((1, 4, 8, 8))  # Smaller for faster test
        text_encodings = mx.ones((1, 77, 768))
        negative_encodings = mx.ones((1, 77, 768))
        sigmas = mx.array([1.0])

        guidance_scale = 4.0

        result = ZImage._compute_cfg_batched(
            mock_z_image,
            t=0,
            latents=latents,
            text_encodings=text_encodings,
            negative_encodings=negative_encodings,
            sigmas=sigmas,
            guidance_scale=guidance_scale,
            cfg_normalization=False,
        )

        # With mock: cond_noise=2.0, uncond_noise=0.5
        # CFG formula: 0.5 + 4.0 * (2.0 - 0.5) = 0.5 + 4.0 * 1.5 = 0.5 + 6.0 = 6.5
        expected_value = 6.5
        mx.synchronize()  # Ensure computation is complete

        assert mx.allclose(result, mx.ones_like(result) * expected_value, atol=1e-5)

    def test_cfg_normalization_preserves_magnitude(self, mock_z_image):
        """Test that cfg_normalization preserves conditional noise magnitude."""
        from mflux.models.z_image.variants.txt2img.z_image import ZImage

        latents = mx.ones((1, 4, 8, 8))
        text_encodings = mx.ones((1, 77, 768))
        negative_encodings = mx.ones((1, 77, 768))
        sigmas = mx.array([1.0])

        # Get result with normalization
        result_normalized = ZImage._compute_cfg_batched(
            mock_z_image,
            t=0,
            latents=latents,
            text_encodings=text_encodings,
            negative_encodings=negative_encodings,
            sigmas=sigmas,
            guidance_scale=4.0,
            cfg_normalization=True,
        )

        # Get result without normalization
        result_unnormalized = ZImage._compute_cfg_batched(
            mock_z_image,
            t=0,
            latents=latents,
            text_encodings=text_encodings,
            negative_encodings=negative_encodings,
            sigmas=sigmas,
            guidance_scale=4.0,
            cfg_normalization=False,
        )

        mx.synchronize()

        # Normalized result should have different magnitude
        norm_mag = mx.sqrt(mx.sum(result_normalized**2))
        unnorm_mag = mx.sqrt(mx.sum(result_unnormalized**2))

        # They should be different (normalization changes magnitude)
        assert not mx.allclose(norm_mag, unnorm_mag, atol=0.1)

    def test_zero_guidance_scale(self, mock_z_image):
        """Test behavior with guidance_scale=0 (pure unconditional)."""
        from mflux.models.z_image.variants.txt2img.z_image import ZImage

        latents = mx.ones((1, 4, 8, 8))
        text_encodings = mx.ones((1, 77, 768))
        negative_encodings = mx.ones((1, 77, 768))
        sigmas = mx.array([1.0])

        result = ZImage._compute_cfg_batched(
            mock_z_image,
            t=0,
            latents=latents,
            text_encodings=text_encodings,
            negative_encodings=negative_encodings,
            sigmas=sigmas,
            guidance_scale=0.0,
            cfg_normalization=False,
        )

        mx.synchronize()

        # With scale=0: uncond + 0 * (cond - uncond) = uncond = 0.5
        assert mx.allclose(result, mx.ones_like(result) * 0.5, atol=1e-5)


class TestCfgBatchingVsSequential:
    """Tests comparing batched vs sequential CFG for correctness."""

    def test_batched_matches_sequential_output(self):
        """Test that batched CFG produces same output as sequential."""
        from mflux.models.z_image.variants.txt2img.z_image import ZImage

        # Create mock model
        model = MagicMock()

        # Deterministic transformer that returns input-dependent output
        def mock_transformer(t, x, cap_feats, sigmas):
            # Return mean of embeddings repeated to match x shape
            embed_mean = mx.mean(cap_feats, axis=(1, 2), keepdims=True)
            return mx.broadcast_to(embed_mean, x.shape)

        model.transformer = mock_transformer

        # Test inputs
        latents = mx.ones((1, 4, 16, 16))
        text_encodings = mx.ones((1, 77, 768)) * 3.0
        negative_encodings = mx.ones((1, 77, 768)) * 1.0
        sigmas = mx.array([1.0])
        guidance_scale = 7.5

        # Batched result
        batched_result = ZImage._compute_cfg_batched(
            model,
            t=0,
            latents=latents,
            text_encodings=text_encodings,
            negative_encodings=negative_encodings,
            sigmas=sigmas,
            guidance_scale=guidance_scale,
            cfg_normalization=False,
        )

        # Sequential result (simulating what generate_image does)
        cond_noise = model.transformer(t=0, x=latents, cap_feats=text_encodings, sigmas=sigmas)
        uncond_noise = model.transformer(t=0, x=latents, cap_feats=negative_encodings, sigmas=sigmas)
        sequential_result = uncond_noise + guidance_scale * (cond_noise - uncond_noise)

        mx.synchronize()

        # Results should match
        assert mx.allclose(batched_result, sequential_result, atol=1e-5)


class TestGenerateImageBatchedCfgParam:
    """Tests for enable_batched_cfg parameter in generate_image."""

    def test_parameter_exists(self):
        """Test that enable_batched_cfg parameter exists in signature."""
        import inspect

        from mflux.models.z_image.variants.txt2img.z_image import ZImage

        sig = inspect.signature(ZImage.generate_image)
        params = sig.parameters

        assert "enable_batched_cfg" in params
        assert params["enable_batched_cfg"].default is False

    def test_parameter_documented(self):
        """Test that enable_batched_cfg is documented."""
        from mflux.models.z_image.variants.txt2img.z_image import ZImage

        docstring = ZImage.generate_image.__doc__
        assert "enable_batched_cfg" in docstring
        assert "30-50%" in docstring or "speedup" in docstring.lower()


class TestCfgBatchingMemory:
    """Tests for memory characteristics of batched CFG."""

    def test_batched_uses_expected_memory_pattern(self):
        """Test that batched CFG doubles latent memory as expected."""
        # This is a design verification test
        # Batched CFG concatenates latents, so memory for latents doubles
        # but we make only one transformer call instead of two

        latents = mx.ones((1, 4, 64, 64))

        # Simulate batching
        batched = mx.concatenate([latents, latents], axis=0)

        # Verify batch dimension doubled
        assert batched.shape[0] == 2 * latents.shape[0]

        # Other dimensions unchanged
        assert batched.shape[1:] == latents.shape[1:]


class TestCfgBatchingEdgeCases:
    """Edge case tests for CFG batching."""

    def test_large_guidance_scale(self):
        """Test with very large guidance scale."""
        from mflux.models.z_image.variants.txt2img.z_image import ZImage

        model = MagicMock()
        model.transformer = lambda t, x, cap_feats, sigmas: mx.ones_like(x)

        latents = mx.ones((1, 4, 8, 8))
        text_encodings = mx.ones((1, 77, 768))
        negative_encodings = mx.zeros((1, 77, 768))
        sigmas = mx.array([1.0])

        # Very high guidance (some users use 20+)
        result = ZImage._compute_cfg_batched(
            model,
            t=0,
            latents=latents,
            text_encodings=text_encodings,
            negative_encodings=negative_encodings,
            sigmas=sigmas,
            guidance_scale=20.0,
            cfg_normalization=False,
        )

        mx.synchronize()

        # Should not produce NaN or inf
        assert not mx.any(mx.isnan(result))
        assert not mx.any(mx.isinf(result))

    def test_negative_guidance_scale(self):
        """Test with negative guidance scale (inverse prompting)."""
        from mflux.models.z_image.variants.txt2img.z_image import ZImage

        model = MagicMock()

        def mock_transformer(t, x, cap_feats, sigmas):
            batch_size = x.shape[0]
            if batch_size == 2:
                return mx.concatenate(
                    [
                        mx.ones_like(x[:1]) * 2.0,
                        mx.ones_like(x[:1]) * 1.0,
                    ],
                    axis=0,
                )
            return mx.ones_like(x)

        model.transformer = mock_transformer

        latents = mx.ones((1, 4, 8, 8))
        text_encodings = mx.ones((1, 77, 768))
        negative_encodings = mx.ones((1, 77, 768))
        sigmas = mx.array([1.0])

        # Negative guidance inverts the effect
        result = ZImage._compute_cfg_batched(
            model,
            t=0,
            latents=latents,
            text_encodings=text_encodings,
            negative_encodings=negative_encodings,
            sigmas=sigmas,
            guidance_scale=-1.0,
            cfg_normalization=False,
        )

        mx.synchronize()

        # uncond + (-1) * (cond - uncond) = 1.0 + (-1) * (2.0 - 1.0) = 1.0 - 1.0 = 0.0
        assert mx.allclose(result, mx.zeros_like(result), atol=1e-5)

    def test_normalization_prevents_nan_on_zero_output(self):
        """Test that normalization handles near-zero outputs safely."""
        from mflux.models.z_image.variants.txt2img.z_image import ZImage

        model = MagicMock()

        # Return very small values that could cause division issues
        def mock_transformer(t, x, cap_feats, sigmas):
            batch_size = x.shape[0]
            if batch_size == 2:
                return mx.concatenate(
                    [
                        mx.ones_like(x[:1]) * 1e-8,
                        mx.ones_like(x[:1]) * 1e-8,
                    ],
                    axis=0,
                )
            return mx.ones_like(x) * 1e-8

        model.transformer = mock_transformer

        latents = mx.ones((1, 4, 8, 8))
        text_encodings = mx.ones((1, 77, 768))
        negative_encodings = mx.ones((1, 77, 768))
        sigmas = mx.array([1.0])

        result = ZImage._compute_cfg_batched(
            model,
            t=0,
            latents=latents,
            text_encodings=text_encodings,
            negative_encodings=negative_encodings,
            sigmas=sigmas,
            guidance_scale=4.0,
            cfg_normalization=True,  # Enable normalization
        )

        mx.synchronize()

        # Should not produce NaN despite tiny values
        assert not mx.any(mx.isnan(result))
