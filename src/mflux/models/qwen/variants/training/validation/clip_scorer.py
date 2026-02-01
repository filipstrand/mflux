"""CLIP-based image-text similarity scoring for Qwen training validation.

Uses a lightweight CLIP model (ViT-B/32) to measure how well generated
images match their prompts. Higher scores indicate better alignment.

Usage:
    scorer = QwenCLIPScorer()  # Lazy loads model on first use
    score = scorer.compute_score(image, "a beautiful sunset")
    print(f"CLIP Score: {score:.2f}")

Score Interpretation:
    80-100: Excellent alignment
    60-80:  Good alignment
    40-60:  Moderate alignment
    <40:    Poor alignment

Training Integration:
    - Use during validation to track model quality over epochs
    - Compare scores before/after training
    - Detect overfitting (training loss decreases but CLIP score drops)

Note:
    The CLIP model is loaded lazily on first use to avoid memory
    overhead when not needed. Use clear() to free memory.
"""

import logging
from typing import TYPE_CHECKING

import mlx.core as mx
import numpy as np

if TYPE_CHECKING:
    from PIL import Image

logger = logging.getLogger(__name__)


class QwenCLIPScorer:
    """CLIP-based image-text similarity scorer for Qwen training.

    Computes cosine similarity between image and text embeddings
    using a CLIP model. Returns scores scaled to 0-100 range.

    The model is loaded lazily on first use and cached for
    subsequent calls. Memory can be freed by calling clear().

    Attributes:
        model_name: HuggingFace model identifier for CLIP
        loaded: Whether the model is currently loaded
    """

    # Default to OpenAI's ViT-B/32 - good balance of speed and quality
    DEFAULT_MODEL = "openai/clip-vit-base-patch32"

    # Score scaling - CLIP cosine similarities are typically 0.2-0.4
    SCORE_SCALE = 100.0
    SCORE_MIN_CLIP = 0.0
    SCORE_MAX_CLIP = 0.5

    def __init__(
        self,
        model_name: str | None = None,
        device: str = "cpu",
    ):
        """Initialize CLIP scorer.

        Args:
            model_name: HuggingFace model identifier. Defaults to ViT-B/32.
            device: Device for computation ("cpu" or "mps"). MLX handles
                   device placement automatically.
        """
        self.model_name = model_name or self.DEFAULT_MODEL
        self.device = device

        # Lazy loading - model loaded on first compute_score call
        self._model = None
        self._processor = None
        self._tokenizer = None

    @property
    def loaded(self) -> bool:
        """Check if model is currently loaded."""
        return self._model is not None

    def _ensure_loaded(self) -> None:
        """Load model if not already loaded.

        Uses transformers library for model loading. The model is
        kept in PyTorch format for compatibility with CLIP.
        """
        if self._model is not None:
            return

        logger.info(f"Loading CLIP model for Qwen training validation: {self.model_name}")

        try:
            from transformers import CLIPModel, CLIPProcessor

            self._processor = CLIPProcessor.from_pretrained(self.model_name)
            self._model = CLIPModel.from_pretrained(self.model_name)
            # Set model to evaluation mode (disables dropout, etc.)
            self._model.eval()

            logger.info("CLIP model loaded successfully")

        except ImportError as e:
            raise ImportError(
                "CLIP scoring requires transformers library. Install with: pip install transformers"
            ) from e
        except Exception as e:
            raise RuntimeError(f"Failed to load CLIP model '{self.model_name}': {e}") from e

    def compute_score(
        self,
        image: "Image.Image | np.ndarray | mx.array",
        prompt: str,
    ) -> float:
        """Compute CLIP similarity score between image and prompt.

        Args:
            image: PIL Image, numpy array, or MLX array
            prompt: Text prompt to compare against

        Returns:
            Similarity score in range [0, 100]. Higher = better alignment.
        """
        self._ensure_loaded()

        pil_image = self._to_pil(image)

        try:
            import torch

            # Type narrowing - _ensure_loaded guarantees these are not None
            processor = self._processor
            model = self._model
            if processor is None or model is None:
                raise RuntimeError("CLIP model not loaded - call _ensure_loaded first")

            inputs = processor(
                text=[prompt],
                images=pil_image,
                return_tensors="pt",
                padding=True,
            )

            with torch.no_grad():
                outputs = model(**inputs)

                image_features = outputs.image_embeds
                text_features = outputs.text_embeds

                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)

                similarity = (image_features @ text_features.T).item()

            score = self._scale_score(similarity)
            return score

        except Exception as e:
            logger.warning(f"CLIP scoring failed: {e}")
            raise

    def compute_scores_batch(
        self,
        images: list["Image.Image | np.ndarray | mx.array"],
        prompts: list[str],
    ) -> list[float]:
        """Compute CLIP scores for multiple image-prompt pairs.

        More efficient than calling compute_score repeatedly.

        Args:
            images: List of images
            prompts: List of prompts (same length as images)

        Returns:
            List of similarity scores in range [0, 100]
        """
        if len(images) != len(prompts):
            raise ValueError(f"Number of images ({len(images)}) must match number of prompts ({len(prompts)})")

        self._ensure_loaded()

        pil_images = [self._to_pil(img) for img in images]

        try:
            import torch

            # Type narrowing - _ensure_loaded guarantees these are not None
            processor = self._processor
            model = self._model
            if processor is None or model is None:
                raise RuntimeError("CLIP model not loaded - call _ensure_loaded first")

            inputs = processor(
                text=prompts,
                images=pil_images,
                return_tensors="pt",
                padding=True,
            )

            with torch.no_grad():
                outputs = model(**inputs)

                image_features = outputs.image_embeds
                text_features = outputs.text_embeds

                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)

                # Pairwise similarities (diagonal of similarity matrix)
                similarities = (image_features * text_features).sum(dim=-1)

            scores = [self._scale_score(sim.item()) for sim in similarities]
            return scores

        except Exception as e:
            logger.warning(f"Batch CLIP scoring failed: {e}")
            raise

    def _to_pil(self, image: "Image.Image | np.ndarray | mx.array") -> "Image.Image":
        """Convert image to PIL format.

        Args:
            image: Input image in various formats. Expected shapes:
                   - (H, W, 3) for RGB
                   - (H, W, 4) for RGBA (alpha will be dropped)
                   - (H, W) for grayscale

        Returns:
            PIL Image in RGB mode

        Raises:
            TypeError: If image type is not supported
            ValueError: If image array is empty or has invalid shape
        """
        from PIL import Image as PILImage

        if isinstance(image, PILImage.Image):
            return image.convert("RGB")

        if isinstance(image, mx.array):
            image = np.array(image)

        if isinstance(image, np.ndarray):
            # Validate array is not empty
            if image.size == 0:
                raise ValueError("Empty image array")

            # Validate shape
            if image.ndim not in (2, 3):
                raise ValueError(f"Expected 2D or 3D array, got shape {image.shape}")
            if image.ndim == 3 and image.shape[2] not in (1, 3, 4):
                raise ValueError(f"Expected 1, 3, or 4 channels, got {image.shape[2]}")

            # Handle dtype conversion
            if image.dtype in (np.float32, np.float64, np.float16):
                # Float images: clamp to [0, 1] then scale to [0, 255]
                image = np.clip(image, 0.0, 1.0)
                image = (image * 255).astype(np.uint8)
            elif image.dtype == np.uint8:
                pass  # Already in correct format
            elif np.issubdtype(image.dtype, np.integer):
                # Other integer types: clamp to [0, 255]
                image = np.clip(image, 0, 255).astype(np.uint8)
            else:
                raise TypeError(f"Unsupported array dtype: {image.dtype}")

            return PILImage.fromarray(image).convert("RGB")

        raise TypeError(f"Unsupported image type: {type(image)}")

    def _scale_score(self, similarity: float) -> float:
        """Scale raw CLIP similarity to 0-100 range.

        Args:
            similarity: Raw cosine similarity from CLIP

        Returns:
            Scaled score in [0, 100]
        """
        clamped = max(self.SCORE_MIN_CLIP, min(similarity, self.SCORE_MAX_CLIP))
        scaled = (clamped - self.SCORE_MIN_CLIP) / (self.SCORE_MAX_CLIP - self.SCORE_MIN_CLIP) * self.SCORE_SCALE
        return float(scaled)

    def clear(self) -> None:
        """Clear loaded model to free memory.

        Call this when done with scoring to release GPU memory.
        The model will be reloaded on the next compute_score call.
        """
        if self._model is not None:
            del self._model
            del self._processor
            self._model = None
            self._processor = None
            logger.info("CLIP model cleared from memory")


def create_qwen_clip_scorer(
    enabled: bool = True,
    model_name: str | None = None,
) -> QwenCLIPScorer | None:
    """Factory function to create CLIP scorer for Qwen training.

    Args:
        enabled: Whether to create scorer (False returns None)
        model_name: Optional custom model name

    Returns:
        QwenCLIPScorer instance or None if disabled
    """
    if not enabled:
        return None

    return QwenCLIPScorer(model_name=model_name)


class QwenTrainingValidator:
    """Combined validation suite for Qwen training.

    Provides comprehensive validation metrics during training:
    - CLIP score (image-text alignment)
    - Diversity metrics (variation across samples)
    - Quality tracking over epochs

    Usage:
        validator = QwenTrainingValidator(enable_clip=True)

        for epoch in range(num_epochs):
            # ... training ...

            # Validation
            images = generate_validation_images(model, val_prompts)
            metrics = validator.validate(images, val_prompts)
            print(f"Epoch {epoch}: CLIP={metrics['clip_mean']:.1f}")

            # Check for quality degradation
            if validator.is_degrading():
                logger.warning("Quality degradation detected!")
    """

    def __init__(
        self,
        enable_clip: bool = True,
        clip_model: str | None = None,
        track_history: bool = True,
        history_window: int = 10,
    ):
        """Initialize training validator.

        Args:
            enable_clip: Whether to compute CLIP scores
            clip_model: Optional custom CLIP model name
            track_history: Whether to track metric history
            history_window: Number of epochs to keep in history
        """
        self.clip_scorer = create_qwen_clip_scorer(enable_clip, clip_model)
        self.track_history = track_history
        self.history_window = history_window

        self._history: list[dict] = []

    def validate(
        self,
        images: list,
        prompts: list[str],
    ) -> dict[str, float]:
        """Run validation on generated images.

        Args:
            images: List of generated images
            prompts: List of prompts used for generation

        Returns:
            Dictionary with validation metrics
        """
        metrics = {}

        # CLIP scores
        if self.clip_scorer is not None:
            try:
                scores = self.clip_scorer.compute_scores_batch(images, prompts)
                metrics["clip_mean"] = np.mean(scores)
                metrics["clip_std"] = np.std(scores)
                metrics["clip_min"] = np.min(scores)
                metrics["clip_max"] = np.max(scores)
            except Exception as e:
                logger.warning(f"CLIP validation failed: {e}")
                metrics["clip_mean"] = -1.0

        # Track history
        if self.track_history:
            self._history.append(metrics)
            if len(self._history) > self.history_window:
                self._history.pop(0)

        return metrics

    def is_degrading(self, threshold: float = -5.0) -> bool:
        """Check if CLIP scores are degrading over recent validation runs.

        Compares the most recent CLIP score to one from 3 runs ago.
        A negative change exceeding the threshold indicates degradation.

        Args:
            threshold: Negative change threshold to trigger warning.
                      Default -5.0 represents 5% of the 0-100 score scale.
                      More negative = more tolerant (e.g., -10.0).
                      Less negative = stricter (e.g., -2.0).

        Returns:
            True if clip_mean has decreased by more than |threshold| points
            over the last 3 validation runs, False otherwise.

        Note:
            Scores are on a 0-100 scale (see SCORE_SCALE).
            A threshold of -5.0 means: trigger if score dropped by >5 points.

        Example:
            If recent scores were [75, 72, 68], change = 68 - 75 = -7.
            With threshold=-5.0, returns True (quality degrading).
        """
        if len(self._history) < 3:
            return False

        recent = [h.get("clip_mean", 0) for h in self._history[-3:]]
        if recent[0] > 0 and recent[-1] > 0:
            change = recent[-1] - recent[0]
            return change < threshold

        return False

    def get_history(self) -> list[dict]:
        """Get validation metric history."""
        return list(self._history)

    def clear(self) -> None:
        """Clear all resources."""
        if self.clip_scorer is not None:
            self.clip_scorer.clear()
        self._history.clear()
