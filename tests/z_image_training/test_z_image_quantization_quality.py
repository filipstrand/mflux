"""Tests for Z-Image quantization quality validation.

These tests verify that quantized models (INT2, INT4, INT8) maintain
acceptable image quality compared to full precision (FP16/BF16).
"""

import numpy as np
import pytest
from PIL import Image

from mflux.utils.image_quality_metrics import ImageQualityMetrics


class TestSSIMComputation:
    """Tests for SSIM metric computation."""

    def test_identical_images_ssim_is_one(self):
        """Test that identical images have SSIM = 1."""
        img = np.random.rand(64, 64, 3)

        ssim = ImageQualityMetrics.compute_ssim_fast(img, img)

        assert abs(ssim - 1.0) < 0.01  # Should be very close to 1

    def test_different_images_ssim_less_than_one(self):
        """Test that different images have SSIM < 1."""
        img1 = np.random.rand(64, 64, 3)
        img2 = np.random.rand(64, 64, 3)

        ssim = ImageQualityMetrics.compute_ssim_fast(img1, img2)

        assert ssim < 1.0

    def test_ssim_is_symmetric(self):
        """Test that SSIM(a, b) == SSIM(b, a)."""
        img1 = np.random.rand(64, 64, 3)
        img2 = img1 + np.random.rand(64, 64, 3) * 0.1

        ssim_forward = ImageQualityMetrics.compute_ssim_fast(img1, img2)
        ssim_reverse = ImageQualityMetrics.compute_ssim_fast(img2, img1)

        assert abs(ssim_forward - ssim_reverse) < 0.01

    def test_ssim_with_pil_images(self):
        """Test SSIM computation with PIL Images."""
        arr = (np.random.rand(64, 64, 3) * 255).astype(np.uint8)
        img1 = Image.fromarray(arr)
        img2 = Image.fromarray(arr)

        ssim = ImageQualityMetrics.compute_ssim_fast(img1, img2)

        assert abs(ssim - 1.0) < 0.01

    def test_ssim_handles_grayscale(self):
        """Test SSIM with grayscale images."""
        img = np.random.rand(64, 64)

        ssim = ImageQualityMetrics.compute_ssim_fast(img, img)

        assert abs(ssim - 1.0) < 0.01

    def test_ssim_shape_mismatch_raises(self):
        """Test that mismatched shapes raise error."""
        img1 = np.random.rand(64, 64, 3)
        img2 = np.random.rand(32, 32, 3)

        with pytest.raises(ValueError, match="same shape"):
            ImageQualityMetrics.compute_ssim_fast(img1, img2)


class TestPSNRComputation:
    """Tests for PSNR metric computation."""

    def test_identical_images_psnr_is_inf(self):
        """Test that identical images have infinite PSNR."""
        img = np.random.rand(64, 64, 3)

        psnr = ImageQualityMetrics.compute_psnr(img, img)

        assert psnr == float("inf")

    def test_different_images_psnr_is_finite(self):
        """Test that different images have finite PSNR."""
        img1 = np.random.rand(64, 64, 3)
        img2 = np.random.rand(64, 64, 3)

        psnr = ImageQualityMetrics.compute_psnr(img1, img2)

        assert psnr < float("inf")
        assert psnr > 0

    def test_small_noise_high_psnr(self):
        """Test that small noise results in high PSNR."""
        img1 = np.random.rand(64, 64, 3)
        # Add very small noise
        img2 = img1 + np.random.rand(64, 64, 3) * 0.001

        psnr = ImageQualityMetrics.compute_psnr(img1, img2)

        # Small noise should give high PSNR (> 50 dB)
        assert psnr > 50

    def test_large_noise_low_psnr(self):
        """Test that large noise results in low PSNR."""
        img1 = np.random.rand(64, 64, 3)
        # Add large noise
        img2 = img1 + np.random.rand(64, 64, 3) * 0.5
        img2 = np.clip(img2, 0, 1)

        psnr = ImageQualityMetrics.compute_psnr(img1, img2)

        # Large noise should give lower PSNR (< 20 dB)
        assert psnr < 20

    def test_psnr_shape_mismatch_raises(self):
        """Test that mismatched shapes raise error."""
        img1 = np.random.rand(64, 64, 3)
        img2 = np.random.rand(32, 32, 3)

        with pytest.raises(ValueError, match="same shape"):
            ImageQualityMetrics.compute_psnr(img1, img2)


class TestMSEComputation:
    """Tests for MSE metric computation."""

    def test_identical_images_mse_is_zero(self):
        """Test that identical images have MSE = 0."""
        img = np.random.rand(64, 64, 3)

        mse = ImageQualityMetrics.compute_mse(img, img)

        assert mse == 0.0

    def test_mse_positive_for_different_images(self):
        """Test that different images have positive MSE."""
        img1 = np.random.rand(64, 64, 3)
        img2 = np.random.rand(64, 64, 3)

        mse = ImageQualityMetrics.compute_mse(img1, img2)

        assert mse > 0


class TestMAEComputation:
    """Tests for MAE metric computation."""

    def test_identical_images_mae_is_zero(self):
        """Test that identical images have MAE = 0."""
        img = np.random.rand(64, 64, 3)

        mae = ImageQualityMetrics.compute_mae(img, img)

        assert mae == 0.0

    def test_mae_positive_for_different_images(self):
        """Test that different images have positive MAE."""
        img1 = np.random.rand(64, 64, 3)
        img2 = np.random.rand(64, 64, 3)

        mae = ImageQualityMetrics.compute_mae(img1, img2)

        assert mae > 0


class TestCompareImages:
    """Tests for compare_images convenience function."""

    def test_returns_all_metrics(self):
        """Test that compare_images returns all expected metrics."""
        img1 = np.random.rand(64, 64, 3)
        img2 = np.random.rand(64, 64, 3)

        metrics = ImageQualityMetrics.compare_images(img1, img2)

        assert "ssim" in metrics
        assert "psnr" in metrics
        assert "mse" in metrics
        assert "mae" in metrics

    def test_identical_images_have_perfect_metrics(self):
        """Test that identical images have perfect metrics."""
        img = np.random.rand(64, 64, 3)

        metrics = ImageQualityMetrics.compare_images(img, img)

        assert abs(metrics["ssim"] - 1.0) < 0.01
        assert metrics["psnr"] == float("inf")
        assert metrics["mse"] == 0.0
        assert metrics["mae"] == 0.0


class TestQuantizationThresholds:
    """Tests for quantization quality threshold checking."""

    def test_thresholds_defined(self):
        """Test that thresholds are defined for supported levels."""
        assert ImageQualityMetrics.SSIM_THRESHOLD_INT8 > 0
        assert ImageQualityMetrics.SSIM_THRESHOLD_INT4 > 0
        assert ImageQualityMetrics.SSIM_THRESHOLD_INT2 > 0

        assert ImageQualityMetrics.PSNR_THRESHOLD_INT8 > 0
        assert ImageQualityMetrics.PSNR_THRESHOLD_INT4 > 0
        assert ImageQualityMetrics.PSNR_THRESHOLD_INT2 > 0

    def test_thresholds_decrease_with_quantization(self):
        """Test that thresholds decrease as quantization increases."""
        # INT8 should have highest threshold (highest quality expected)
        assert ImageQualityMetrics.SSIM_THRESHOLD_INT8 > ImageQualityMetrics.SSIM_THRESHOLD_INT4
        assert ImageQualityMetrics.SSIM_THRESHOLD_INT4 > ImageQualityMetrics.SSIM_THRESHOLD_INT2

        assert ImageQualityMetrics.PSNR_THRESHOLD_INT8 > ImageQualityMetrics.PSNR_THRESHOLD_INT4
        assert ImageQualityMetrics.PSNR_THRESHOLD_INT4 > ImageQualityMetrics.PSNR_THRESHOLD_INT2

    def test_passes_quality_threshold_int8(self):
        """Test quality threshold checking for INT8."""
        # Good quality metrics for INT8
        good_metrics = {"ssim": 0.96, "psnr": 36.0}
        passes, _ = ImageQualityMetrics.passes_quality_threshold(good_metrics, 8)
        assert passes is True

        # Poor quality metrics for INT8
        poor_metrics = {"ssim": 0.90, "psnr": 30.0}
        passes, reason = ImageQualityMetrics.passes_quality_threshold(poor_metrics, 8)
        assert passes is False
        assert "SSIM" in reason

    def test_passes_quality_threshold_int4(self):
        """Test quality threshold checking for INT4."""
        # Good quality metrics for INT4
        good_metrics = {"ssim": 0.92, "psnr": 32.0}
        passes, _ = ImageQualityMetrics.passes_quality_threshold(good_metrics, 4)
        assert passes is True

        # Poor quality metrics for INT4
        poor_metrics = {"ssim": 0.85, "psnr": 25.0}
        passes, reason = ImageQualityMetrics.passes_quality_threshold(poor_metrics, 4)
        assert passes is False

    def test_passes_quality_threshold_int2(self):
        """Test quality threshold checking for INT2."""
        # Good quality metrics for INT2
        good_metrics = {"ssim": 0.82, "psnr": 27.0}
        passes, _ = ImageQualityMetrics.passes_quality_threshold(good_metrics, 2)
        assert passes is True

        # Poor quality metrics for INT2
        poor_metrics = {"ssim": 0.70, "psnr": 20.0}
        passes, reason = ImageQualityMetrics.passes_quality_threshold(poor_metrics, 2)
        assert passes is False

    def test_unknown_quantization_level_passes(self):
        """Test that unknown quantization levels pass by default."""
        metrics = {"ssim": 0.5, "psnr": 10.0}

        passes, reason = ImageQualityMetrics.passes_quality_threshold(metrics, 6)

        assert passes is True
        assert "No thresholds" in reason


class TestSimulatedQuantizationDegradation:
    """Tests simulating quality degradation from quantization."""

    def test_small_noise_passes_int8(self):
        """Test that small noise (simulating INT8) passes threshold."""
        img1 = np.random.rand(64, 64, 3)
        # INT8 typically adds very small error
        img2 = img1 + np.random.randn(64, 64, 3) * 0.005
        img2 = np.clip(img2, 0, 1)

        metrics = ImageQualityMetrics.compare_images(img1, img2)
        passes, _ = ImageQualityMetrics.passes_quality_threshold(metrics, 8)

        # Small noise should pass INT8 threshold
        assert passes is True

    def test_medium_noise_passes_int4(self):
        """Test that medium noise (simulating INT4) passes threshold."""
        img1 = np.random.rand(64, 64, 3)
        # INT4 adds moderate error
        img2 = img1 + np.random.randn(64, 64, 3) * 0.02
        img2 = np.clip(img2, 0, 1)

        metrics = ImageQualityMetrics.compare_images(img1, img2)
        passes, _ = ImageQualityMetrics.passes_quality_threshold(metrics, 4)

        # Medium noise should pass INT4 threshold
        assert passes is True

    def test_large_noise_passes_int2(self):
        """Test that larger noise (simulating INT2) passes threshold."""
        img1 = np.random.rand(64, 64, 3)
        # INT2 adds more significant error
        img2 = img1 + np.random.randn(64, 64, 3) * 0.05
        img2 = np.clip(img2, 0, 1)

        metrics = ImageQualityMetrics.compare_images(img1, img2)
        passes, _ = ImageQualityMetrics.passes_quality_threshold(metrics, 2)

        # Larger noise should still pass INT2 threshold (more lenient)
        assert passes is True


class TestEdgeCases:
    """Edge case tests for image quality metrics."""

    def test_all_zeros_images(self):
        """Test metrics with all-zero images."""
        img = np.zeros((64, 64, 3))

        metrics = ImageQualityMetrics.compare_images(img, img)

        # Should handle without errors
        assert metrics["mse"] == 0.0
        assert metrics["mae"] == 0.0

    def test_all_ones_images(self):
        """Test metrics with all-one images."""
        img = np.ones((64, 64, 3))

        metrics = ImageQualityMetrics.compare_images(img, img)

        # Should handle without errors
        assert metrics["mse"] == 0.0
        assert metrics["mae"] == 0.0

    def test_high_contrast_images(self):
        """Test metrics with high contrast images."""
        # Checkerboard pattern
        img1 = np.zeros((64, 64, 3))
        img1[::2, ::2, :] = 1.0
        img1[1::2, 1::2, :] = 1.0

        img2 = 1.0 - img1  # Inverted checkerboard

        metrics = ImageQualityMetrics.compare_images(img1, img2)

        # Inverted images should have very low SSIM
        assert metrics["ssim"] < 0.1

    def test_small_image(self):
        """Test metrics with very small images."""
        img = np.random.rand(8, 8, 3)

        metrics = ImageQualityMetrics.compare_images(img, img)

        # Should handle small images
        assert abs(metrics["ssim"] - 1.0) < 0.1

    def test_uint8_input(self):
        """Test metrics with uint8 input (0-255 range)."""
        arr = (np.random.rand(64, 64, 3) * 255).astype(np.uint8)

        metrics = ImageQualityMetrics.compare_images(arr, arr)

        # Should normalize correctly
        assert abs(metrics["ssim"] - 1.0) < 0.01
