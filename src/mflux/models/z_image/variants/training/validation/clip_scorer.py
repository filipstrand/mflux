"""CLIP-based image-text similarity scoring for training validation.

Uses a lightweight CLIP model (ViT-B/32) to measure how well generated
images match their prompts. Higher scores indicate better alignment.

Usage:
    scorer = CLIPScorer()  # Lazy loads model on first use
    score = scorer.compute_score(image, "a beautiful sunset")
    print(f"CLIP Score: {score:.2f}")

Score Interpretation:
    80-100: Excellent alignment
    60-80:  Good alignment
    40-60:  Moderate alignment
    <40:    Poor alignment

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


class CLIPScorer:
    """CLIP-based image-text similarity scorer.

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
    # We scale to 0-100 for more intuitive interpretation
    SCORE_SCALE = 100.0
    SCORE_MIN_CLIP = 0.0  # Minimum expected CLIP similarity
    SCORE_MAX_CLIP = 0.5  # Maximum expected CLIP similarity (rare)

    def __init__(
        self,
        model_name: str | None = None,
        device: str = "cpu",
    ):
        """Initialize CLIP scorer.

        Args:
            model_name: HuggingFace model identifier. Defaults to ViT-B/32.
            device: Device for computation ("cpu" or "mps"). MLX handles
                   device placement automatically, so this is mainly for
                   compatibility.
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
        converted to MLX arrays for efficient computation.
        """
        if self._model is not None:
            return

        logger.info(f"Loading CLIP model: {self.model_name}")

        try:
            # Import here to avoid dependency at module load
            from transformers import CLIPModel, CLIPProcessor

            # Load model and processor
            self._processor = CLIPProcessor.from_pretrained(self.model_name)
            self._model = CLIPModel.from_pretrained(self.model_name)

            # Move to evaluation mode
            self._model.eval_mode = True

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
        assert self._processor is not None, "_ensure_loaded should have loaded processor"
        assert self._model is not None, "_ensure_loaded should have loaded model"

        # Convert image to PIL if needed
        pil_image = self._to_pil(image)

        try:
            import torch

            # Process inputs
            inputs = self._processor(
                text=[prompt],
                images=pil_image,
                return_tensors="pt",
                padding=True,
            )

            # Get embeddings (no gradients needed)
            with torch.no_grad():
                outputs = self._model(**inputs)

                # Get normalized features
                image_features = outputs.image_embeds
                text_features = outputs.text_embeds

                # Normalize
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)

                # Cosine similarity
                similarity = (image_features @ text_features.T).item()

            # Scale to 0-100 range
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

        Args:
            images: List of images
            prompts: List of prompts (same length as images)

        Returns:
            List of similarity scores in range [0, 100]
        """
        if len(images) != len(prompts):
            raise ValueError(f"Number of images ({len(images)}) must match number of prompts ({len(prompts)})")

        self._ensure_loaded()
        assert self._processor is not None, "_ensure_loaded should have loaded processor"
        assert self._model is not None, "_ensure_loaded should have loaded model"

        # Convert all images to PIL
        pil_images = [self._to_pil(img) for img in images]

        try:
            import torch

            # Process all inputs at once
            inputs = self._processor(
                text=prompts,
                images=pil_images,
                return_tensors="pt",
                padding=True,
            )

            with torch.no_grad():
                outputs = self._model(**inputs)

                # Get normalized features
                image_features = outputs.image_embeds
                text_features = outputs.text_embeds

                # Normalize
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)

                # Pairwise similarities (diagonal of similarity matrix)
                similarities = (image_features * text_features).sum(dim=-1)

            # Scale scores
            scores = [self._scale_score(sim.item()) for sim in similarities]

            return scores

        except Exception as e:
            logger.warning(f"Batch CLIP scoring failed: {e}")
            raise

    def _to_pil(self, image: "Image.Image | np.ndarray | mx.array") -> "Image.Image":
        """Convert image to PIL format.

        Args:
            image: Input image in various formats

        Returns:
            PIL Image in RGB mode
        """
        from PIL import Image as PILImage

        if isinstance(image, PILImage.Image):
            return image.convert("RGB")

        if isinstance(image, mx.array):
            image = np.array(image)

        if isinstance(image, np.ndarray):
            # Handle various numpy formats
            if image.dtype == np.float32 or image.dtype == np.float64:
                # Assume [0, 1] range
                image = (image * 255).astype(np.uint8)
            elif image.max() <= 1.0:
                image = (image * 255).astype(np.uint8)

            return PILImage.fromarray(image).convert("RGB")

        raise TypeError(f"Unsupported image type: {type(image)}")

    def _scale_score(self, similarity: float) -> float:
        """Scale raw CLIP similarity to 0-100 range.

        Raw CLIP similarities are typically in [0.15, 0.35] range.
        We scale to [0, 100] for more intuitive interpretation.

        Args:
            similarity: Raw cosine similarity from CLIP

        Returns:
            Scaled score in [0, 100]
        """
        # Clamp to expected range
        clamped = max(self.SCORE_MIN_CLIP, min(similarity, self.SCORE_MAX_CLIP))

        # Linear scale to [0, 100]
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


def create_clip_scorer(
    enabled: bool = True,
    model_name: str | None = None,
) -> CLIPScorer | None:
    """Factory function to create CLIP scorer.

    Args:
        enabled: Whether to create scorer (False returns None)
        model_name: Optional custom model name

    Returns:
        CLIPScorer instance or None if disabled
    """
    if not enabled:
        return None

    return CLIPScorer(model_name=model_name)
