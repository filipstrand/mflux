"""Native MLX Qwen3-VL Embedding model.

Provides embedding extraction for text, images, and mixed content
using the Qwen3-VL-Embedding-2B model on Apple Silicon.

Target: 5-10x speedup over PyTorch implementation with quality parity.
"""

import logging
from dataclasses import dataclass
from typing import Any

import mlx.core as mx
from mlx import nn

from .pooling import normalize_embeddings, pool_last_token
from .qwen3_vl_2b_encoder import Qwen3VL2BEncoder

logger = logging.getLogger(__name__)


# Qwen3-VL Vision Tokenization Constants
# The model uses 14x14 pixel patches with 2x2 spatial merging:
#   - IMAGE_BASE_FACTOR = 16 (base patch size)
#   - IMAGE_FACTOR = 32 (effective patch size after 2x spatial merge)
# Min/max pixels define the token budget for image encoding:
#   - MIN_PIXELS ~4 vision tokens minimum
#   - MAX_PIXELS ~1800 vision tokens maximum
IMAGE_BASE_FACTOR = 16
IMAGE_FACTOR = IMAGE_BASE_FACTOR * 2  # 32 after spatial merge
MIN_PIXELS = 4 * IMAGE_FACTOR * IMAGE_FACTOR  # 4096 pixels = ~4 tokens
MAX_PIXELS = 1800 * IMAGE_FACTOR * IMAGE_FACTOR  # 1843200 pixels = ~1800 tokens
MAX_LENGTH = 8192  # Maximum sequence length in tokens


@dataclass
class Qwen3VLEmbeddingConfig:
    """Configuration for Qwen3-VL Embedding model."""

    # Model architecture (2B variant)
    vocab_size: int = 152064
    hidden_size: int = 2048
    num_hidden_layers: int = 28
    num_attention_heads: int = 16
    num_key_value_heads: int = 4
    intermediate_size: int = 8192
    max_position_embeddings: int = 128000
    rope_theta: float = 1000000.0
    rms_norm_eps: float = 1e-6

    # Vision configuration
    vision_embed_dim: int = 1280
    vision_depth: int = 32
    vision_num_heads: int = 16

    # Inference configuration
    max_length: int = MAX_LENGTH
    min_pixels: int = MIN_PIXELS
    max_pixels: int = MAX_PIXELS


class Qwen3VLEmbedding(nn.Module):
    """Qwen3-VL Embedding model for computing embeddings.

    This model extracts embeddings from text, images, or mixed content
    using last-token pooling followed by L2 normalization.
    """

    def __init__(self, config: Qwen3VLEmbeddingConfig):
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

    def __call__(
        self,
        input_ids: mx.array,
        attention_mask: mx.array,
        pixel_values: mx.array | None = None,
        image_grid_thw: mx.array | None = None,
        normalize: bool = True,
    ) -> mx.array:
        """Compute embeddings for the input.

        Args:
            input_ids: Token IDs [batch, seq_len]
            attention_mask: Attention mask [batch, seq_len]
            pixel_values: Optional preprocessed image pixels
            image_grid_thw: Optional image grid dimensions
            normalize: Whether to L2 normalize the embeddings

        Returns:
            Embeddings [batch, hidden_size]
        """
        # Forward through encoder
        hidden_states = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
        )

        # Pool last token
        embeddings = pool_last_token(hidden_states, attention_mask)

        # Normalize if requested
        if normalize:
            embeddings = normalize_embeddings(embeddings)

        return embeddings


class Qwen3VLEmbedder:
    """High-level embedder matching the official Qwen3VLEmbedder interface.

    This class provides a user-friendly interface for computing embeddings
    from text, images, or mixed content.

    Usage:
        embedder = Qwen3VLEmbedder.from_pretrained("Qwen/Qwen3-VL-Embedding-2B")

        # Text embedding
        text_emb = embedder.process([{"text": "hello", "instruction": "..."}])

        # Image embedding
        img_emb = embedder.process([{"image": pil_image, "instruction": "..."}])

        # Mixed embedding
        mixed_emb = embedder.process([{"text": "...", "image": img, "instruction": "..."}])
    """

    DEFAULT_MODEL = "Qwen/Qwen3-VL-Embedding-2B"
    DEFAULT_INSTRUCTION = "Represent the user's input."

    def __init__(
        self,
        model: Qwen3VLEmbedding,
        tokenizer: Any,
        processor: Any,
        config: Qwen3VLEmbeddingConfig,
    ):
        """Initialize the embedder.

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
    ) -> "Qwen3VLEmbedder":
        """Load a pretrained embedder.

        Args:
            model_name_or_path: HuggingFace model ID or local path
            quantize_vision: Whether to quantize vision encoder (INT8)

        Returns:
            Initialized Qwen3VLEmbedder
        """
        from .weights import EmbeddingWeightHandler

        logger.info(f"Loading Qwen3VL Embedding model: {model_name_or_path}")

        # Load config and create model
        config = Qwen3VLEmbeddingConfig()
        model = Qwen3VLEmbedding(config)

        # Load weights
        weight_handler = EmbeddingWeightHandler(model_name_or_path)
        weight_handler.load_weights(model, quantize_vision=quantize_vision)

        # Initialize vision after weights are loaded
        model.encoder.init_vision()

        # Load tokenizer and processor
        try:
            from transformers import AutoProcessor, AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
            processor = AutoProcessor.from_pretrained(model_name_or_path)
        except ImportError:
            raise ImportError("transformers library required for tokenization. Install with: pip install transformers")

        logger.info("Qwen3VL Embedding model loaded successfully")
        return cls(model, tokenizer, processor, config)

    def compile(self) -> None:
        """Compile the model for faster inference.

        Enables MLX graph compilation for 15-40% speedup.
        """
        if not self._compiled:
            self.model.__call__ = mx.compile(self.model.__call__)
            self._compiled = True
            logger.info("Model compiled for faster inference")

    def process(
        self,
        inputs: list[dict[str, Any]],
        normalize: bool = True,
    ) -> mx.array:
        """Process inputs and return embeddings.

        Args:
            inputs: List of input dictionaries with optional keys:
                - text: Text content
                - image: PIL Image or path
                - instruction: Instruction prefix
            normalize: Whether to L2 normalize embeddings

        Returns:
            Embeddings array [batch, hidden_size]
        """
        # Format inputs into conversations
        conversations = [self._format_input(inp) for inp in inputs]

        # Preprocess with tokenizer
        processed = self._preprocess(conversations)

        # Forward through model
        embeddings = self.model(
            input_ids=processed["input_ids"],
            attention_mask=processed["attention_mask"],
            pixel_values=processed.get("pixel_values"),
            image_grid_thw=processed.get("image_grid_thw"),
            normalize=normalize,
        )

        # Force evaluation (MLX lazy evaluation)
        mx.synchronize()

        return embeddings

    def _format_input(self, inp: dict[str, Any]) -> list[dict]:
        """Format a single input into conversation format."""
        instruction = inp.get("instruction", self.DEFAULT_INSTRUCTION)
        if instruction and instruction[-1] not in ".!?":
            instruction = instruction + "."

        content = []
        conversation = [
            {"role": "system", "content": [{"type": "text", "text": instruction}]},
            {"role": "user", "content": content},
        ]

        # Add image if present
        if "image" in inp:
            image = inp["image"]
            content.append(
                {
                    "type": "image",
                    "image": image,
                    "min_pixels": self.config.min_pixels,
                    "max_pixels": self.config.max_pixels,
                }
            )

        # Add text if present
        if "text" in inp:
            content.append({"type": "text", "text": inp["text"]})

        # Default to NULL if no content
        if not content:
            content.append({"type": "text", "text": "NULL"})

        return conversation

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

        # Tokenize
        inputs = self.processor(
            text=text,
            images=images,
            videos=videos,
            video_metadata=video_metadata,
            truncation=True,
            max_length=self.config.max_length,
            padding=True,
            do_resize=False,
            return_tensors="np",
            **video_kwargs,
        )

        # Convert to MLX arrays
        result = {
            "input_ids": mx.array(inputs["input_ids"]),
            "attention_mask": mx.array(inputs["attention_mask"]),
        }

        if "pixel_values" in inputs and inputs["pixel_values"] is not None:
            result["pixel_values"] = mx.array(inputs["pixel_values"])
        if "image_grid_thw" in inputs and inputs["image_grid_thw"] is not None:
            result["image_grid_thw"] = mx.array(inputs["image_grid_thw"])

        return result

    def clear(self) -> None:
        """Release model resources."""
        del self.model
        del self.tokenizer
        del self.processor
        self.model = None
        self.tokenizer = None
        self.processor = None
