"""Tests for Z-Image CLIP scoring functionality.

CLIP scoring measures prompt-image alignment during training validation.
"""

import inspect
from unittest.mock import MagicMock, patch

import mlx.core as mx
import numpy as np
import pytest
from PIL import Image


class TestCLIPScorerInit:
    """Tests for CLIPScorer initialization."""

    def test_class_exists(self):
        """Test that CLIPScorer class exists."""
        from mflux.models.z_image.variants.training.validation.clip_scorer import (
            CLIPScorer,
        )

        assert CLIPScorer is not None

    def test_default_model_name(self):
        """Test default model name is set."""
        from mflux.models.z_image.variants.training.validation.clip_scorer import (
            CLIPScorer,
        )

        scorer = CLIPScorer()

        assert scorer.model_name == CLIPScorer.DEFAULT_MODEL
        assert "clip" in scorer.model_name.lower()

    def test_custom_model_name(self):
        """Test custom model name is accepted."""
        from mflux.models.z_image.variants.training.validation.clip_scorer import (
            CLIPScorer,
        )

        custom_model = "custom/clip-model"
        scorer = CLIPScorer(model_name=custom_model)

        assert scorer.model_name == custom_model

    def test_lazy_loading(self):
        """Test that model is not loaded on init."""
        from mflux.models.z_image.variants.training.validation.clip_scorer import (
            CLIPScorer,
        )

        scorer = CLIPScorer()

        assert scorer.loaded is False
        assert scorer._model is None


class TestCLIPScorerScaling:
    """Tests for score scaling logic."""

    def test_scale_constants_defined(self):
        """Test that scaling constants are defined."""
        from mflux.models.z_image.variants.training.validation.clip_scorer import (
            CLIPScorer,
        )

        assert CLIPScorer.SCORE_SCALE == 100.0
        assert CLIPScorer.SCORE_MIN_CLIP >= 0.0
        assert CLIPScorer.SCORE_MAX_CLIP > CLIPScorer.SCORE_MIN_CLIP

    def test_scale_score_min(self):
        """Test scaling at minimum value."""
        from mflux.models.z_image.variants.training.validation.clip_scorer import (
            CLIPScorer,
        )

        scorer = CLIPScorer()

        score = scorer._scale_score(CLIPScorer.SCORE_MIN_CLIP)

        assert score == 0.0

    def test_scale_score_max(self):
        """Test scaling at maximum value."""
        from mflux.models.z_image.variants.training.validation.clip_scorer import (
            CLIPScorer,
        )

        scorer = CLIPScorer()

        score = scorer._scale_score(CLIPScorer.SCORE_MAX_CLIP)

        assert score == 100.0

    def test_scale_score_mid(self):
        """Test scaling at midpoint."""
        from mflux.models.z_image.variants.training.validation.clip_scorer import (
            CLIPScorer,
        )

        scorer = CLIPScorer()
        mid = (CLIPScorer.SCORE_MIN_CLIP + CLIPScorer.SCORE_MAX_CLIP) / 2

        score = scorer._scale_score(mid)

        assert abs(score - 50.0) < 0.1

    def test_scale_score_clamps_below_min(self):
        """Test that values below min are clamped."""
        from mflux.models.z_image.variants.training.validation.clip_scorer import (
            CLIPScorer,
        )

        scorer = CLIPScorer()

        score = scorer._scale_score(-0.5)

        assert score == 0.0

    def test_scale_score_clamps_above_max(self):
        """Test that values above max are clamped."""
        from mflux.models.z_image.variants.training.validation.clip_scorer import (
            CLIPScorer,
        )

        scorer = CLIPScorer()

        score = scorer._scale_score(1.0)

        assert score == 100.0


class TestCLIPScorerImageConversion:
    """Tests for image conversion utility."""

    def test_pil_image_passthrough(self):
        """Test PIL image passes through correctly."""
        from mflux.models.z_image.variants.training.validation.clip_scorer import (
            CLIPScorer,
        )

        scorer = CLIPScorer()
        pil_img = Image.new("RGB", (64, 64), color="red")

        result = scorer._to_pil(pil_img)

        assert isinstance(result, Image.Image)
        assert result.mode == "RGB"

    def test_pil_image_converts_mode(self):
        """Test non-RGB PIL image is converted."""
        from mflux.models.z_image.variants.training.validation.clip_scorer import (
            CLIPScorer,
        )

        scorer = CLIPScorer()
        pil_img = Image.new("L", (64, 64), color=128)  # Grayscale

        result = scorer._to_pil(pil_img)

        assert result.mode == "RGB"

    def test_numpy_float_array(self):
        """Test numpy float array conversion."""
        from mflux.models.z_image.variants.training.validation.clip_scorer import (
            CLIPScorer,
        )

        scorer = CLIPScorer()
        arr = np.random.rand(64, 64, 3).astype(np.float32)

        result = scorer._to_pil(arr)

        assert isinstance(result, Image.Image)
        assert result.mode == "RGB"
        assert result.size == (64, 64)

    def test_numpy_uint8_array(self):
        """Test numpy uint8 array conversion."""
        from mflux.models.z_image.variants.training.validation.clip_scorer import (
            CLIPScorer,
        )

        scorer = CLIPScorer()
        arr = (np.random.rand(64, 64, 3) * 255).astype(np.uint8)

        result = scorer._to_pil(arr)

        assert isinstance(result, Image.Image)
        assert result.mode == "RGB"

    def test_mlx_array(self):
        """Test MLX array conversion."""
        from mflux.models.z_image.variants.training.validation.clip_scorer import (
            CLIPScorer,
        )

        scorer = CLIPScorer()
        arr = mx.random.uniform(shape=(64, 64, 3))

        result = scorer._to_pil(arr)

        assert isinstance(result, Image.Image)
        assert result.mode == "RGB"

    def test_unsupported_type_raises(self):
        """Test that unsupported types raise error."""
        from mflux.models.z_image.variants.training.validation.clip_scorer import (
            CLIPScorer,
        )

        scorer = CLIPScorer()

        with pytest.raises(TypeError, match="Unsupported"):
            scorer._to_pil("not an image")


class TestCLIPScorerComputeScore:
    """Tests for compute_score method."""

    def test_method_signature(self):
        """Test compute_score method signature."""
        from mflux.models.z_image.variants.training.validation.clip_scorer import (
            CLIPScorer,
        )

        sig = inspect.signature(CLIPScorer.compute_score)
        params = list(sig.parameters.keys())

        assert "self" in params
        assert "image" in params
        assert "prompt" in params

    def test_requires_transformers(self):
        """Test that missing transformers raises ImportError."""
        from mflux.models.z_image.variants.training.validation.clip_scorer import (
            CLIPScorer,
        )

        scorer = CLIPScorer()

        # Mock import error
        with patch.dict("sys.modules", {"transformers": None}):
            with patch(
                "builtins.__import__",
                side_effect=ImportError("No module named 'transformers'"),
            ):
                with pytest.raises(ImportError, match="transformers"):
                    scorer._ensure_loaded()


class TestCLIPScorerComputeScoresBatch:
    """Tests for compute_scores_batch method."""

    def test_method_exists(self):
        """Test compute_scores_batch method exists."""
        from mflux.models.z_image.variants.training.validation.clip_scorer import (
            CLIPScorer,
        )

        assert hasattr(CLIPScorer, "compute_scores_batch")

    def test_validates_length_mismatch(self):
        """Test that mismatched lengths raise error."""
        from mflux.models.z_image.variants.training.validation.clip_scorer import (
            CLIPScorer,
        )

        scorer = CLIPScorer()
        scorer._model = MagicMock()  # Prevent loading
        scorer._processor = MagicMock()

        images = [Image.new("RGB", (64, 64)) for _ in range(3)]
        prompts = ["a", "b"]  # Different length

        with pytest.raises(ValueError, match="must match"):
            scorer.compute_scores_batch(images, prompts)


class TestCLIPScorerClear:
    """Tests for clear method."""

    def test_clear_unloads_model(self):
        """Test that clear unloads the model."""
        from mflux.models.z_image.variants.training.validation.clip_scorer import (
            CLIPScorer,
        )

        scorer = CLIPScorer()
        scorer._model = MagicMock()
        scorer._processor = MagicMock()

        assert scorer.loaded is True

        scorer.clear()

        assert scorer.loaded is False
        assert scorer._model is None
        assert scorer._processor is None

    def test_clear_when_not_loaded(self):
        """Test that clear is safe when not loaded."""
        from mflux.models.z_image.variants.training.validation.clip_scorer import (
            CLIPScorer,
        )

        scorer = CLIPScorer()

        # Should not raise
        scorer.clear()

        assert scorer.loaded is False


class TestCreateClipScorer:
    """Tests for factory function."""

    def test_create_enabled(self):
        """Test creating enabled scorer."""
        from mflux.models.z_image.variants.training.validation.clip_scorer import (
            CLIPScorer,
            create_clip_scorer,
        )

        scorer = create_clip_scorer(enabled=True)

        assert isinstance(scorer, CLIPScorer)

    def test_create_disabled(self):
        """Test that disabled returns None."""
        from mflux.models.z_image.variants.training.validation.clip_scorer import (
            create_clip_scorer,
        )

        scorer = create_clip_scorer(enabled=False)

        assert scorer is None

    def test_create_custom_model(self):
        """Test creating with custom model."""
        from mflux.models.z_image.variants.training.validation.clip_scorer import (
            create_clip_scorer,
        )

        custom = "custom/model"
        scorer = create_clip_scorer(enabled=True, model_name=custom)

        assert scorer.model_name == custom


class TestCLIPScorerInterpretation:
    """Tests for score interpretation guidelines."""

    def test_score_ranges_documented(self):
        """Test that score interpretation ranges are in docstring."""
        from mflux.models.z_image.variants.training.validation.clip_scorer import (
            CLIPScorer,
        )

        docstring = CLIPScorer.__doc__

        # Should mention score ranges
        assert "80" in docstring or "Excellent" in docstring
        assert "alignment" in docstring.lower()

    def test_compute_score_returns_float(self):
        """Test that compute_score returns float type."""
        from mflux.models.z_image.variants.training.validation.clip_scorer import (
            CLIPScorer,
        )

        scorer = CLIPScorer()

        # Test scale_score returns float
        result = scorer._scale_score(0.25)

        assert isinstance(result, float)

    def test_scores_in_expected_range(self):
        """Test that scaled scores are in [0, 100]."""
        from mflux.models.z_image.variants.training.validation.clip_scorer import (
            CLIPScorer,
        )

        scorer = CLIPScorer()

        # Test various raw similarity values
        test_values = [-0.5, 0.0, 0.1, 0.25, 0.35, 0.5, 1.0]

        for val in test_values:
            score = scorer._scale_score(val)
            assert 0.0 <= score <= 100.0, f"Score {score} out of range for input {val}"
