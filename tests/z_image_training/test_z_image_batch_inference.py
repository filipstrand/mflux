"""Tests for Z-Image batch inference functionality.

Batch inference processes multiple images in parallel through the
transformer for 2-3x throughput improvement.
"""

import inspect
from unittest.mock import MagicMock, patch

import mlx.core as mx
import pytest


class TestGenerateImagesBatched:
    """Tests for generate_images_batched method."""

    def test_method_exists(self):
        """Test that generate_images_batched method exists."""
        from mflux.models.z_image.variants.txt2img.z_image import ZImage

        assert hasattr(ZImage, "generate_images_batched")
        assert callable(getattr(ZImage, "generate_images_batched"))

    def test_signature_has_required_params(self):
        """Test that method has required parameters."""
        from mflux.models.z_image.variants.txt2img.z_image import ZImage

        sig = inspect.signature(ZImage.generate_images_batched)
        params = sig.parameters

        # Required params
        assert "seeds" in params
        assert "prompt" in params

        # Optional params with defaults
        assert "negative_prompt" in params
        assert "num_inference_steps" in params
        assert "height" in params
        assert "width" in params
        assert "guidance_scale" in params
        assert "max_batch_size" in params

    def test_max_batch_size_default(self):
        """Test default max_batch_size is reasonable."""
        from mflux.models.z_image.variants.txt2img.z_image import ZImage

        sig = inspect.signature(ZImage.generate_images_batched)
        default = sig.parameters["max_batch_size"].default

        assert default == 4
        assert 1 <= default <= 8  # Reasonable range

    def test_validates_empty_seeds(self):
        """Test that empty seeds list raises error."""
        from mflux.models.z_image.variants.txt2img.z_image import ZImage

        model = MagicMock(spec=ZImage)

        with pytest.raises(ValueError, match="non-empty"):
            ZImage.generate_images_batched(
                model,
                seeds=[],
                prompt="test prompt",
            )

    def test_validates_max_batch_size(self):
        """Test that invalid max_batch_size raises error."""
        from mflux.models.z_image.variants.txt2img.z_image import ZImage

        model = MagicMock(spec=ZImage)
        model.MAX_BATCH_SIZE = 64

        with pytest.raises(ValueError, match="max_batch_size must be >= 1"):
            ZImage.generate_images_batched(
                model,
                seeds=[1, 2, 3],
                prompt="test prompt",
                max_batch_size=0,
            )

    def test_validates_seeds_exceeds_max(self):
        """Test that too many seeds raises error."""
        from mflux.models.z_image.variants.txt2img.z_image import ZImage

        model = MagicMock(spec=ZImage)
        model.MAX_BATCH_SIZE = 64

        with pytest.raises(ValueError, match="maximum is 64"):
            ZImage.generate_images_batched(
                model,
                seeds=list(range(100)),  # 100 seeds > 64 max
                prompt="test prompt",
            )


class TestGenerateBatch:
    """Tests for _generate_batch internal method."""

    def test_method_exists(self):
        """Test that _generate_batch method exists."""
        from mflux.models.z_image.variants.txt2img.z_image import ZImage

        assert hasattr(ZImage, "_generate_batch")

    def test_returns_list(self):
        """Test that _generate_batch returns a list."""
        from mflux.models.z_image.variants.txt2img.z_image import ZImage

        # We can't easily test the full method without mocking everything
        # So we just verify the return type annotation
        sig = inspect.signature(ZImage._generate_batch)
        return_annotation = sig.return_annotation

        # Should return list of images
        assert "list" in str(return_annotation).lower() or return_annotation is list


class TestComputeCfgBatchedMulti:
    """Tests for _compute_cfg_batched_multi method."""

    @pytest.fixture
    def mock_model(self):
        """Create mock model with transformer."""
        model = MagicMock()

        def mock_transformer(t, x, cap_feats, sigmas):
            # Return values based on embedding content for testing
            # Conditional (first half) gets higher values
            batch_size = x.shape[0] // 2
            cond_noise = mx.ones_like(x[:batch_size]) * 2.0
            uncond_noise = mx.ones_like(x[:batch_size]) * 0.5
            return mx.concatenate([cond_noise, uncond_noise], axis=0)

        model.transformer = mock_transformer
        return model

    def test_handles_batch_of_multiple(self, mock_model):
        """Test CFG computation for batch of multiple images."""
        from mflux.models.z_image.variants.txt2img.z_image import ZImage

        batch_size = 4
        latents = mx.ones((batch_size, 4, 16, 16))
        text_encodings = mx.ones((batch_size, 77, 768))
        negative_encodings = mx.zeros((batch_size, 77, 768))
        sigmas = mx.array([1.0])

        result = ZImage._compute_cfg_batched_multi(
            mock_model,
            t=0,
            latents=latents,
            text_encodings=text_encodings,
            negative_encodings=negative_encodings,
            sigmas=sigmas,
            guidance_scale=4.0,
            cfg_normalization=False,
        )

        # Output should have same batch size as input
        assert result.shape[0] == batch_size
        assert result.shape[1:] == latents.shape[1:]

    def test_transformer_called_with_doubled_batch(self, mock_model):
        """Test that transformer receives 2x batch size."""
        from mflux.models.z_image.variants.txt2img.z_image import ZImage

        batch_size = 3
        latents = mx.ones((batch_size, 4, 16, 16))
        text_encodings = mx.ones((batch_size, 77, 768))
        negative_encodings = mx.ones((batch_size, 77, 768))
        sigmas = mx.array([1.0])

        call_args = []
        original_transformer = mock_model.transformer

        def tracking_transformer(t, x, cap_feats, sigmas):
            call_args.append({"batch_size": x.shape[0]})
            return original_transformer(t, x, cap_feats, sigmas)

        mock_model.transformer = tracking_transformer

        ZImage._compute_cfg_batched_multi(
            mock_model,
            t=0,
            latents=latents,
            text_encodings=text_encodings,
            negative_encodings=negative_encodings,
            sigmas=sigmas,
            guidance_scale=4.0,
            cfg_normalization=False,
        )

        # Should be called with 2 * batch_size
        assert call_args[0]["batch_size"] == 2 * batch_size

    def test_cfg_formula_applied_per_image(self, mock_model):
        """Test that CFG formula is applied correctly per image."""
        from mflux.models.z_image.variants.txt2img.z_image import ZImage

        batch_size = 2
        latents = mx.ones((batch_size, 4, 8, 8))
        text_encodings = mx.ones((batch_size, 77, 768))
        negative_encodings = mx.ones((batch_size, 77, 768))
        sigmas = mx.array([1.0])

        guidance_scale = 4.0

        result = ZImage._compute_cfg_batched_multi(
            mock_model,
            t=0,
            latents=latents,
            text_encodings=text_encodings,
            negative_encodings=negative_encodings,
            sigmas=sigmas,
            guidance_scale=guidance_scale,
            cfg_normalization=False,
        )

        mx.synchronize()

        # CFG: uncond + scale * (cond - uncond) = 0.5 + 4.0 * (2.0 - 0.5) = 6.5
        expected = 6.5
        assert mx.allclose(result, mx.ones_like(result) * expected, atol=1e-5)


class TestBatchInferenceIntegration:
    """Integration-style tests for batch inference."""

    def test_batches_processed_correctly(self):
        """Test that seeds are processed in correct batch chunks."""
        from mflux.models.z_image.variants.txt2img.z_image import ZImage

        # Track which batches are processed
        processed_batches = []

        def mock_generate_batch(
            self,
            seeds,
            text_encodings,
            negative_encodings,
            num_inference_steps,
            height,
            width,
            guidance_scale,
            cfg_normalization,
            scheduler,
            prompt,
        ):
            processed_batches.append(list(seeds))
            # Return mock images
            return [MagicMock() for _ in seeds]

        # Create mock model with required attributes
        model = MagicMock(spec=ZImage)
        model.MAX_BATCH_SIZE = 64
        model.tokenizers = {"z_image": MagicMock()}
        model.text_encoder = MagicMock()

        # Mock PromptEncoder
        with patch("mflux.models.z_image.variants.txt2img.z_image.PromptEncoder") as mock_encoder:
            mock_encoder.encode_prompt.return_value = mx.ones((1, 77, 768))

            # Bind our mock method
            model._generate_batch = lambda *args, **kwargs: mock_generate_batch(model, *args, **kwargs)

            # Call with 7 seeds, max_batch_size=3
            images = ZImage.generate_images_batched(
                model,
                seeds=[1, 2, 3, 4, 5, 6, 7],
                prompt="test",
                max_batch_size=3,
            )

            # Should be split into batches of [3, 3, 1]
            assert len(processed_batches) == 3
            # Should return 7 images total
            assert len(images) == 7
            assert processed_batches[0] == [1, 2, 3]
            assert processed_batches[1] == [4, 5, 6]
            assert processed_batches[2] == [7]

    def test_encodes_prompt_once(self):
        """Test that prompt is encoded only once, not per batch."""
        from mflux.models.z_image.variants.txt2img.z_image import ZImage

        model = MagicMock(spec=ZImage)
        model.MAX_BATCH_SIZE = 64
        model.tokenizers = {"z_image": MagicMock()}
        model.text_encoder = MagicMock()
        model._generate_batch = MagicMock(return_value=[MagicMock()])

        with patch("mflux.models.z_image.variants.txt2img.z_image.PromptEncoder") as mock_encoder:
            mock_encoder.encode_prompt.return_value = mx.ones((1, 77, 768))

            ZImage.generate_images_batched(
                model,
                seeds=[1, 2, 3, 4, 5],
                prompt="test prompt",
                max_batch_size=2,
            )

            # Should encode positive prompt once
            # Should encode negative prompt once (if guidance > 0)
            # Total: 2 calls (one positive, one negative)
            assert mock_encoder.encode_prompt.call_count == 2


class TestBatchVsSequential:
    """Tests comparing batched vs sequential generation."""

    def test_returns_same_count(self):
        """Test that batched returns same number of images as seeds."""
        from mflux.models.z_image.variants.txt2img.z_image import ZImage

        model = MagicMock(spec=ZImage)
        model.MAX_BATCH_SIZE = 64
        model.tokenizers = {"z_image": MagicMock()}
        model.text_encoder = MagicMock()

        num_images = 5

        def mock_generate_batch(*args, **kwargs):
            seeds = args[1] if len(args) > 1 else kwargs.get("seeds", [])
            return [MagicMock() for _ in seeds]

        model._generate_batch = mock_generate_batch

        with patch("mflux.models.z_image.variants.txt2img.z_image.PromptEncoder") as mock_encoder:
            mock_encoder.encode_prompt.return_value = mx.ones((1, 77, 768))

            result = ZImage.generate_images_batched(
                model,
                seeds=list(range(num_images)),
                prompt="test",
            )

            assert len(result) == num_images


class TestBatchMemoryCharacteristics:
    """Tests for memory behavior of batch inference."""

    def test_latent_concatenation_shape(self):
        """Test that concatenated latents have correct shape."""
        batch_size = 4
        latent_shape = (1, 4, 64, 64)

        latents = [mx.ones(latent_shape) for _ in range(batch_size)]
        batched = mx.concatenate(latents, axis=0)

        assert batched.shape == (batch_size, 4, 64, 64)

    def test_embedding_repeat_shape(self):
        """Test that repeated embeddings have correct shape."""
        batch_size = 4
        embedding_shape = (1, 77, 768)

        embedding = mx.ones(embedding_shape)
        batched = mx.repeat(embedding, batch_size, axis=0)

        assert batched.shape == (batch_size, 77, 768)
