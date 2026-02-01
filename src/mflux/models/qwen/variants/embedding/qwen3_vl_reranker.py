"""Native MLX Qwen3-VL Reranker model.

Provides relevance scoring between queries and documents (text, images, or mixed)
using the Qwen3-VL-Reranker-2B model on Apple Silicon.

The reranker uses a binary classification approach:
- Extracts the last token hidden state
- Projects through a score weight derived from (yes - no) token embeddings
- Applies sigmoid to get a [0, 1] relevance score

Target: 5-10x speedup over PyTorch implementation with quality parity.
"""

import logging
from dataclasses import dataclass
from typing import Any

import mlx.core as mx
from mlx import nn

from .pooling import pool_last_token
from .qwen3_vl_2b_encoder import Qwen3VL2BEncoder

logger = logging.getLogger(__name__)


# Constants from the official implementation
IMAGE_BASE_FACTOR = 16
IMAGE_FACTOR = IMAGE_BASE_FACTOR * 2
MIN_PIXELS = 4 * IMAGE_FACTOR * IMAGE_FACTOR
MAX_PIXELS = 1800 * IMAGE_FACTOR * IMAGE_FACTOR
MAX_LENGTH = 10240  # Reranker uses longer context


@dataclass
class Qwen3VLRerankerConfig:
    """Configuration for Qwen3-VL Reranker model."""

    # Model architecture (2B variant - from HuggingFace weights)
    vocab_size: int = 151936  # Actual from embed_tokens.weight
    hidden_size: int = 2048
    num_hidden_layers: int = 28
    num_attention_heads: int = 16
    num_key_value_heads: int = 8  # 8, not 4 - from k_proj shape
    intermediate_size: int = 8192
    max_position_embeddings: int = 128000
    rope_theta: float = 1000000.0
    rms_norm_eps: float = 1e-6

    # Inference configuration
    max_length: int = MAX_LENGTH
    min_pixels: int = MIN_PIXELS
    max_pixels: int = MAX_PIXELS


class Qwen3VLReranking(nn.Module):
    """Qwen3-VL Reranker model for computing relevance scores.

    Uses binary classification approach with (yes - no) score weight.
    """

    def __init__(self, config: Qwen3VLRerankerConfig):
        super().__init__()
        self.config = config

        # Initialize encoder
        self.encoder = Qwen3VL2BEncoder(
            vocab_size=config.vocab_size,
            hidden_size=config.hidden_size,
            num_hidden_layers=config.num_hidden_layers,
            num_attention_heads=config.num_attention_heads,
            num_key_value_heads=config.num_key_value_heads,
            intermediate_size=config.intermediate_size,
            max_position_embeddings=config.max_position_embeddings,
            rope_theta=config.rope_theta,
            rms_norm_eps=config.rms_norm_eps,
            include_vision=True,
        )

        # Score weight: (yes - no) token embedding difference
        # This is loaded from the language model head
        self.score_weight = None

    def set_score_weight(self, weight: mx.array) -> None:
        """Set the binary classification score weight.

        Args:
            weight: Score weight [hidden_size], derived from lm_head[yes] - lm_head[no]
        """
        self.score_weight = weight

    def __call__(
        self,
        input_ids: mx.array,
        attention_mask: mx.array,
        pixel_values: mx.array | None = None,
        image_grid_thw: mx.array | None = None,
    ) -> mx.array:
        """Compute relevance scores for query-document pairs.

        Args:
            input_ids: Token IDs [batch, seq_len]
            attention_mask: Attention mask [batch, seq_len]
            pixel_values: Optional preprocessed image pixels
            image_grid_thw: Optional image grid dimensions

        Returns:
            Scores [batch] in range [0, 1]
        """
        if self.score_weight is None:
            raise RuntimeError("Score weight not set. Call set_score_weight() first.")

        # Forward through encoder
        hidden_states = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
        )

        # Pool last token
        last_hidden = pool_last_token(hidden_states, attention_mask)

        # Compute scores using dot product with score weight
        scores = mx.sum(last_hidden * self.score_weight, axis=-1)

        # Apply sigmoid for [0, 1] output
        scores = mx.sigmoid(scores)

        return scores


class Qwen3VLReranker:
    """High-level reranker matching the official Qwen3VLReranker interface.

    This class provides a user-friendly interface for computing relevance scores
    between queries and documents containing text, images, or mixed content.

    Usage:
        reranker = Qwen3VLReranker.from_pretrained("Qwen/Qwen3-VL-Reranker-2B")

        inputs = {
            "instruction": "Retrieve images matching the query.",
            "query": {"text": "a beautiful sunset"},
            "documents": [
                {"image": pil_image1},
                {"image": pil_image2},
            ]
        }
        scores = reranker.process(inputs)
        # Returns: [0.85, 0.23] - relevance scores for each document
    """

    DEFAULT_MODEL = "Qwen/Qwen3-VL-Reranker-2B"
    DEFAULT_INSTRUCTION = "Given a search query, retrieve relevant candidates that answer the query."

    def __init__(
        self,
        model: Qwen3VLReranking,
        tokenizer: Any,
        processor: Any,
        config: Qwen3VLRerankerConfig,
    ):
        """Initialize the reranker.

        Use from_pretrained() to create an instance with loaded weights.
        """
        self.model = model
        self.tokenizer = tokenizer
        self.processor = processor
        self.config = config
        self._compiled = False

    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: str = DEFAULT_MODEL,
        quantize_vision: bool = False,
    ) -> "Qwen3VLReranker":
        """Load a pretrained reranker.

        Args:
            model_name_or_path: HuggingFace model ID or local path
            quantize_vision: Whether to quantize vision encoder (INT8)

        Returns:
            Initialized Qwen3VLReranker
        """
        from .weights import EmbeddingWeightHandler

        logger.info(f"Loading Qwen3VL Reranker model: {model_name_or_path}")

        # Load config and create model
        config = Qwen3VLRerankerConfig()
        model = Qwen3VLReranking(config)

        # Load weights (including score weight)
        weight_handler = EmbeddingWeightHandler(model_name_or_path)
        weight_handler.load_weights(model, quantize_vision=quantize_vision, is_reranker=True)

        # Initialize vision after weights are loaded
        model.encoder.init_vision()

        # Load tokenizer and processor
        try:
            from transformers import AutoProcessor, AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
            processor = AutoProcessor.from_pretrained(model_name_or_path, padding_side="left")
        except ImportError:
            raise ImportError("transformers library required for tokenization. Install with: pip install transformers")

        logger.info("Qwen3VL Reranker model loaded successfully")
        return cls(model, tokenizer, processor, config)

    def compile(self) -> None:
        """Compile the model for faster inference."""
        if not self._compiled:
            self.model.__call__ = mx.compile(self.model.__call__)
            self._compiled = True
            logger.info("Model compiled for faster inference")

    def process(self, inputs: dict[str, Any]) -> list[float]:
        """Process query-document pairs and return relevance scores.

        Args:
            inputs: Dictionary with:
                - instruction: Optional instruction prefix
                - query: Query dict with text/image/video
                - documents: List of document dicts with text/image/video

        Returns:
            List of relevance scores [0, 1] for each document
        """
        instruction = inputs.get("instruction", self.DEFAULT_INSTRUCTION)
        query = inputs.get("query", {})
        documents = inputs.get("documents", [])

        if not query or not documents:
            return []

        # Format each query-document pair
        pairs = [self._format_pair(query, doc, instruction) for doc in documents]

        # Compute scores for each pair
        scores = []
        for pair in pairs:
            processed = self._preprocess([pair])
            score = self.model(
                input_ids=processed["input_ids"],
                attention_mask=processed["attention_mask"],
                pixel_values=processed.get("pixel_values"),
                image_grid_thw=processed.get("image_grid_thw"),
            )
            mx.synchronize()
            scores.append(float(score[0]))

        return scores

    def _format_pair(
        self,
        query: dict[str, Any],
        document: dict[str, Any],
        instruction: str,
    ) -> list[dict]:
        """Format a query-document pair for scoring."""
        contents = []

        # System message
        system_prompt = (
            "Judge whether the Document meets the requirements based on the Query "
            'and the Instruct provided. Note that the answer can only be "yes" or "no".'
        )

        # Add instruction
        contents.append({"type": "text", "text": f"<Instruct>: {instruction}"})

        # Add query content
        query_content = self._format_content(query, prefix="<Query>:")
        contents.extend(query_content)

        # Add document content
        doc_content = self._format_content(document, prefix="\n<Document>:")
        contents.extend(doc_content)

        return [
            {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
            {"role": "user", "content": contents},
        ]

    def _format_content(
        self,
        content: dict[str, Any],
        prefix: str,
    ) -> list[dict[str, Any]]:
        """Format content with optional text/image/video."""
        result: list[dict[str, Any]] = [{"type": "text", "text": prefix}]

        # Add image if present
        if "image" in content:
            result.append(
                {
                    "type": "image",
                    "image": content["image"],
                    "min_pixels": self.config.min_pixels,
                    "max_pixels": self.config.max_pixels,
                }
            )

        # Add text if present
        if "text" in content:
            result.append({"type": "text", "text": content["text"]})

        # Default to NULL if no content beyond prefix
        if len(result) == 1:
            result.append({"type": "text", "text": "NULL"})

        return result

    def _preprocess(self, conversations: list[list[dict]]) -> dict[str, mx.array]:
        """Preprocess conversations into model inputs."""
        try:
            from qwen_vl_utils.vision_process import process_vision_info
        except ImportError:
            raise ImportError(
                "qwen-vl-utils required for image processing. Install with: pip install qwen-vl-utils>=0.0.14"
            )

        # Apply chat template
        text = self.processor.apply_chat_template(
            conversations,
            add_generation_prompt=True,
            tokenize=False,
        )

        # Process vision info
        try:
            images, video_inputs, video_kwargs = process_vision_info(
                conversations,
                image_patch_size=16,
                return_video_metadata=True,
                return_video_kwargs=True,
            )
        except (ValueError, TypeError, OSError) as e:
            logger.warning(f"Vision processing failed: {e}")
            images = None
            video_inputs = None
            video_kwargs = {"do_sample_frames": False}

        # Handle video inputs
        if video_inputs is not None:
            videos, video_metadata = zip(*video_inputs)
            videos = list(videos)
            video_metadata = list(video_metadata)
        else:
            videos, video_metadata = None, None

        # Tokenize with truncation handling (must use "pt" as processor doesn't support "np" directly)
        inputs = self.processor(
            text=text,
            images=images,
            videos=videos,
            video_metadata=video_metadata,
            truncation=True,
            max_length=self.config.max_length,
            padding=True,
            do_resize=False,
            return_tensors="pt",
            **video_kwargs,
        )

        # Convert PyTorch tensors to MLX arrays via numpy
        result = {
            "input_ids": mx.array(inputs["input_ids"].numpy()),
            "attention_mask": mx.array(inputs["attention_mask"].numpy()),
        }

        if "pixel_values" in inputs and inputs["pixel_values"] is not None:
            result["pixel_values"] = mx.array(inputs["pixel_values"].numpy())
        if "image_grid_thw" in inputs and inputs["image_grid_thw"] is not None:
            result["image_grid_thw"] = mx.array(inputs["image_grid_thw"].numpy())

        return result

    def clear(self) -> None:
        """Release model resources."""
        del self.model
        del self.tokenizer
        del self.processor
        self.model = None
        self.tokenizer = None
        self.processor = None
