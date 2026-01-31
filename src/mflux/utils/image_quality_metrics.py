"""Image quality metrics for quantization validation.

Provides SSIM, PSNR, and simplified perceptual metrics for comparing
images generated with different quantization levels.

Quality Thresholds (recommended):
    INT8: SSIM >= 0.95, PSNR >= 35dB
    INT4: SSIM >= 0.90, PSNR >= 30dB
    INT2: SSIM >= 0.80, PSNR >= 25dB

Usage:
    from mflux.utils.image_quality_metrics import ImageQualityMetrics

    metrics = ImageQualityMetrics.compare_images(image_fp32, image_int4)
    print(f"SSIM: {metrics['ssim']:.4f}")
    print(f"PSNR: {metrics['psnr']:.2f} dB")
"""

import math
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from PIL import Image


class ImageQualityMetrics:
    """Image quality comparison utilities.

    All methods accept either PIL Images or numpy arrays.
    """

    # Quality thresholds for different quantization levels
    SSIM_THRESHOLD_INT8 = 0.95
    SSIM_THRESHOLD_INT4 = 0.90
    SSIM_THRESHOLD_INT2 = 0.80

    PSNR_THRESHOLD_INT8 = 35.0  # dB
    PSNR_THRESHOLD_INT4 = 30.0
    PSNR_THRESHOLD_INT2 = 25.0

    @staticmethod
    def _to_array(image: "Image.Image | np.ndarray") -> np.ndarray:
        """Convert PIL Image or array to numpy float array [0, 1].

        Args:
            image: PIL Image or numpy array

        Returns:
            Numpy array with values in [0, 1]
        """
        if hasattr(image, "convert"):
            # PIL Image
            image = np.array(image.convert("RGB"))

        image = np.asarray(image, dtype=np.float64)

        # Normalize to [0, 1] if needed
        if image.max() > 1.0:
            image = image / 255.0

        return image

    @staticmethod
    def compute_ssim(
        image1: "Image.Image | np.ndarray",
        image2: "Image.Image | np.ndarray",
        window_size: int = 11,
        k1: float = 0.01,
        k2: float = 0.03,
    ) -> float:
        """Compute Structural Similarity Index (SSIM) between two images.

        SSIM measures perceived quality by comparing luminance, contrast,
        and structure between images. Values range from -1 to 1, with
        1 indicating identical images.

        Args:
            image1: Reference image
            image2: Comparison image
            window_size: Size of sliding window (default: 11)
            k1: Stability constant for luminance (default: 0.01)
            k2: Stability constant for contrast (default: 0.03)

        Returns:
            SSIM value in range [-1, 1], typically [0, 1] for similar images

        References:
            Wang, Z., et al. "Image quality assessment: from error
            visibility to structural similarity." IEEE TIP, 2004.
        """
        img1 = ImageQualityMetrics._to_array(image1)
        img2 = ImageQualityMetrics._to_array(image2)

        if img1.shape != img2.shape:
            raise ValueError(f"Images must have same shape. Got {img1.shape} and {img2.shape}")

        # Constants
        L = 1.0  # Dynamic range for [0, 1] images
        c1 = (k1 * L) ** 2
        c2 = (k2 * L) ** 2

        # Convert to grayscale if RGB (use luminance weights)
        if len(img1.shape) == 3 and img1.shape[2] == 3:
            weights = np.array([0.2989, 0.5870, 0.1140])
            img1 = np.dot(img1, weights)
            img2 = np.dot(img2, weights)

        # Create Gaussian window
        sigma = 1.5
        gauss = np.exp(-(np.arange(-(window_size // 2), window_size // 2 + 1) ** 2) / (2 * sigma**2))
        gauss = gauss / gauss.sum()
        window = np.outer(gauss, gauss)

        # Pad images
        pad = window_size // 2
        img1_padded = np.pad(img1, pad, mode="reflect")
        img2_padded = np.pad(img2, pad, mode="reflect")

        # Compute local statistics using convolution
        # For simplicity, we use a sliding window approach
        h, w = img1.shape
        ssim_map = np.zeros((h, w))

        for i in range(h):
            for j in range(w):
                patch1 = img1_padded[i : i + window_size, j : j + window_size]
                patch2 = img2_padded[i : i + window_size, j : j + window_size]

                # Weighted means
                mu1 = np.sum(window * patch1)
                mu2 = np.sum(window * patch2)

                # Weighted variances and covariance
                sigma1_sq = np.sum(window * (patch1 - mu1) ** 2)
                sigma2_sq = np.sum(window * (patch2 - mu2) ** 2)
                sigma12 = np.sum(window * (patch1 - mu1) * (patch2 - mu2))

                # SSIM formula
                numerator = (2 * mu1 * mu2 + c1) * (2 * sigma12 + c2)
                denominator = (mu1**2 + mu2**2 + c1) * (sigma1_sq + sigma2_sq + c2)
                ssim_map[i, j] = numerator / denominator

        return float(np.mean(ssim_map))

    @staticmethod
    def compute_ssim_fast(
        image1: "Image.Image | np.ndarray",
        image2: "Image.Image | np.ndarray",
    ) -> float:
        """Fast approximate SSIM computation.

        Uses block-based computation for speed. Less accurate than
        full SSIM but much faster for large images.

        Args:
            image1: Reference image
            image2: Comparison image

        Returns:
            Approximate SSIM value
        """
        img1 = ImageQualityMetrics._to_array(image1)
        img2 = ImageQualityMetrics._to_array(image2)

        if img1.shape != img2.shape:
            raise ValueError(f"Images must have same shape. Got {img1.shape} and {img2.shape}")

        # Convert to grayscale
        if len(img1.shape) == 3:
            weights = np.array([0.2989, 0.5870, 0.1140])
            img1 = np.dot(img1, weights)
            img2 = np.dot(img2, weights)

        # Block-based SSIM (8x8 blocks)
        block_size = 8
        k1, k2 = 0.01, 0.03
        L = 1.0
        c1 = (k1 * L) ** 2
        c2 = (k2 * L) ** 2

        h, w = img1.shape
        n_blocks_h = h // block_size
        n_blocks_w = w // block_size

        ssim_values = []
        for i in range(n_blocks_h):
            for j in range(n_blocks_w):
                y = i * block_size
                x = j * block_size
                block1 = img1[y : y + block_size, x : x + block_size]
                block2 = img2[y : y + block_size, x : x + block_size]

                mu1 = np.mean(block1)
                mu2 = np.mean(block2)
                sigma1_sq = np.var(block1)
                sigma2_sq = np.var(block2)
                sigma12 = np.mean((block1 - mu1) * (block2 - mu2))

                numerator = (2 * mu1 * mu2 + c1) * (2 * sigma12 + c2)
                denominator = (mu1**2 + mu2**2 + c1) * (sigma1_sq + sigma2_sq + c2)
                ssim_values.append(numerator / denominator)

        return float(np.mean(ssim_values))

    @staticmethod
    def compute_psnr(
        image1: "Image.Image | np.ndarray",
        image2: "Image.Image | np.ndarray",
    ) -> float:
        """Compute Peak Signal-to-Noise Ratio (PSNR) between two images.

        PSNR measures the ratio between the maximum possible signal
        power and the power of corrupting noise. Higher values indicate
        better quality. Typical values: 30-50 dB for good quality.

        Args:
            image1: Reference image
            image2: Comparison image

        Returns:
            PSNR in decibels (dB). Returns float('inf') for identical images.
        """
        img1 = ImageQualityMetrics._to_array(image1)
        img2 = ImageQualityMetrics._to_array(image2)

        if img1.shape != img2.shape:
            raise ValueError(f"Images must have same shape. Got {img1.shape} and {img2.shape}")

        # Mean Squared Error
        mse = np.mean((img1 - img2) ** 2)

        if mse == 0:
            return float("inf")

        # PSNR formula: 10 * log10(MAX^2 / MSE)
        max_pixel = 1.0  # Images are normalized to [0, 1]
        psnr = 10 * math.log10(max_pixel**2 / mse)

        return float(psnr)

    @staticmethod
    def compute_mse(
        image1: "Image.Image | np.ndarray",
        image2: "Image.Image | np.ndarray",
    ) -> float:
        """Compute Mean Squared Error between two images.

        Args:
            image1: Reference image
            image2: Comparison image

        Returns:
            MSE value (lower is better, 0 = identical)
        """
        img1 = ImageQualityMetrics._to_array(image1)
        img2 = ImageQualityMetrics._to_array(image2)

        if img1.shape != img2.shape:
            raise ValueError(f"Images must have same shape. Got {img1.shape} and {img2.shape}")

        return float(np.mean((img1 - img2) ** 2))

    @staticmethod
    def compute_mae(
        image1: "Image.Image | np.ndarray",
        image2: "Image.Image | np.ndarray",
    ) -> float:
        """Compute Mean Absolute Error between two images.

        Args:
            image1: Reference image
            image2: Comparison image

        Returns:
            MAE value (lower is better, 0 = identical)
        """
        img1 = ImageQualityMetrics._to_array(image1)
        img2 = ImageQualityMetrics._to_array(image2)

        if img1.shape != img2.shape:
            raise ValueError(f"Images must have same shape. Got {img1.shape} and {img2.shape}")

        return float(np.mean(np.abs(img1 - img2)))

    @staticmethod
    def compare_images(
        image1: "Image.Image | np.ndarray",
        image2: "Image.Image | np.ndarray",
        fast_ssim: bool = True,
    ) -> dict:
        """Compute multiple quality metrics between two images.

        Args:
            image1: Reference image (typically from higher precision model)
            image2: Comparison image (typically from quantized model)
            fast_ssim: Use fast approximate SSIM (default: True)

        Returns:
            Dictionary with keys: ssim, psnr, mse, mae
        """
        ssim_fn = ImageQualityMetrics.compute_ssim_fast if fast_ssim else ImageQualityMetrics.compute_ssim

        return {
            "ssim": ssim_fn(image1, image2),
            "psnr": ImageQualityMetrics.compute_psnr(image1, image2),
            "mse": ImageQualityMetrics.compute_mse(image1, image2),
            "mae": ImageQualityMetrics.compute_mae(image1, image2),
        }

    @staticmethod
    def passes_quality_threshold(
        metrics: dict,
        quantization_bits: int,
    ) -> tuple[bool, str]:
        """Check if metrics pass quality threshold for quantization level.

        Args:
            metrics: Dictionary with 'ssim' and 'psnr' keys
            quantization_bits: Quantization level (2, 4, or 8)

        Returns:
            Tuple of (passes: bool, reason: str)
        """
        thresholds = {
            8: (ImageQualityMetrics.SSIM_THRESHOLD_INT8, ImageQualityMetrics.PSNR_THRESHOLD_INT8),
            4: (ImageQualityMetrics.SSIM_THRESHOLD_INT4, ImageQualityMetrics.PSNR_THRESHOLD_INT4),
            2: (ImageQualityMetrics.SSIM_THRESHOLD_INT2, ImageQualityMetrics.PSNR_THRESHOLD_INT2),
        }

        if quantization_bits not in thresholds:
            return True, f"No thresholds defined for {quantization_bits}-bit"

        ssim_thresh, psnr_thresh = thresholds[quantization_bits]
        ssim = metrics.get("ssim", 0)
        psnr = metrics.get("psnr", 0)

        failures = []
        if ssim < ssim_thresh:
            failures.append(f"SSIM {ssim:.4f} < {ssim_thresh}")
        if psnr < psnr_thresh:
            failures.append(f"PSNR {psnr:.2f}dB < {psnr_thresh}dB")

        if failures:
            return False, "; ".join(failures)

        return True, "Quality acceptable"
