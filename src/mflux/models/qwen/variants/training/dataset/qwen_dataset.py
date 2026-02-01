"""
Qwen-Image Training Dataset.

Handles dataset preparation including:
- Image encoding via VAE
- Text encoding via Qwen text encoder
- Data augmentation
- Embedding caching for 2-3x speedup
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mlx.core as mx
import PIL.Image
from mlx import nn
from tqdm import tqdm

from mflux.models.qwen.latent_creator.qwen_latent_creator import QwenLatentCreator
from mflux.models.qwen.variants.training.dataset.qwen_batch import QwenExample
from mflux.models.qwen.variants.training.dataset.qwen_preprocessing import QwenPreprocessing
from mflux.models.qwen.variants.training.optimization.embedding_cache import EmbeddingCache
from mflux.utils.image_util import ImageUtil


@dataclass
class QwenExampleSpec:
    """Specification for a training example (before encoding)."""

    prompt: str
    image: Path | str

    def __post_init__(self):
        self.image = Path(self.image)


class QwenDataset:
    """
    Dataset for Qwen-Image training.

    Handles encoding of images and text, with optional caching
    for significant speedup during training.

    Attributes:
        examples: List of encoded QwenExample objects
    """

    def __init__(self, examples: list[QwenExample]):
        self.examples = examples

    @staticmethod
    def prepare_dataset(
        qwen: Any,  # QwenImage model
        raw_data: list[QwenExampleSpec],
        width: int,
        height: int,
        augment: bool = True,
        cache: EmbeddingCache | None = None,
    ) -> "QwenDataset":
        """
        Prepare dataset by encoding all examples.

        Args:
            qwen: QwenImage model (for VAE and text encoder)
            raw_data: List of (prompt, image_path) specifications
            width: Target image width
            height: Target image height
            augment: Whether to apply data augmentation
            cache: Optional embedding cache for speedup

        Returns:
            Prepared QwenDataset
        """
        # Encode all examples
        examples = QwenDataset._create_examples(
            qwen=qwen,
            raw_data=raw_data,
            width=width,
            height=height,
            cache=cache,
        )

        # Apply augmentation if enabled
        if augment:
            augmented = []
            for example in examples:
                augmented.extend(QwenPreprocessing.augment(example))
            examples = augmented

        return QwenDataset(examples)

    @staticmethod
    def _create_examples(
        qwen: Any,
        raw_data: list[QwenExampleSpec],
        width: int,
        height: int,
        cache: EmbeddingCache | None = None,
    ) -> list[QwenExample]:
        """
        Create encoded examples from raw data.

        Uses caching if available for 2-3x speedup.
        """
        examples = []

        for i, entry in enumerate(tqdm(raw_data, desc="Encoding dataset")):
            # Encode image (with optional caching)
            if cache is not None:
                encoded_image = cache.get_image_latent(
                    entry.image,
                    vae=qwen.vae,
                    compute_fn=lambda p, v: QwenDataset._encode_image(v, p, width=width, height=height),
                )
            else:
                encoded_image = QwenDataset._encode_image(qwen.vae, entry.image, width=width, height=height)

            # Encode prompt (with optional caching)
            if cache is not None:
                prompt_embeds, prompt_mask = cache.get_text_embedding(
                    entry.prompt,
                    encoder=qwen.text_encoder,
                    tokenizer=qwen.tokenizers["qwen"],
                    compute_fn=lambda p, e, t: QwenDataset._encode_prompt(e, t, p),
                )
            else:
                prompt_embeds, prompt_mask = QwenDataset._encode_prompt(
                    qwen.text_encoder, qwen.tokenizers["qwen"], entry.prompt
                )

            # Create example
            example = QwenExample(
                example_id=i,
                prompt=entry.prompt,
                image_path=entry.image,
                encoded_image=encoded_image,
                prompt_embeds=prompt_embeds,
                prompt_mask=prompt_mask,
            )
            examples.append(example)

            # Force evaluation to enable progress tracking
            mx.eval(encoded_image, prompt_embeds, prompt_mask)

        return examples

    @staticmethod
    def _encode_image(
        vae: nn.Module,
        image_path: Path | str,
        width: int,
        height: int,
    ) -> mx.array:
        """
        Encode image to latent space.

        Args:
            vae: QwenVAE model
            image_path: Path to image file
            width: Target width
            height: Target height

        Returns:
            Packed latent tensor
        """
        image = PIL.Image.open(Path(image_path).resolve()).convert("RGB")
        scaled_image = ImageUtil.scale_to_dimensions(image, target_width=width, target_height=height)
        encoded = vae.encode(ImageUtil.to_array(scaled_image))
        latents = QwenLatentCreator.pack_latents(encoded, width=width, height=height)
        return latents

    @staticmethod
    def _encode_prompt(
        text_encoder: Any,
        tokenizer: Any,
        prompt: str,
    ) -> tuple[mx.array, mx.array]:
        """
        Encode text prompt.

        Args:
            text_encoder: Qwen text encoder
            tokenizer: Qwen tokenizer
            prompt: Text prompt

        Returns:
            Tuple of (embeddings, attention_mask)
        """
        # Tokenize
        tokens = tokenizer.tokenize(prompt)

        # Encode
        embeds = text_encoder(tokens.input_ids)

        # Create attention mask (1 for real tokens, 0 for padding)
        mask = mx.ones(tokens.input_ids.shape, dtype=mx.int32)

        return embeds, mask

    def size(self) -> int:
        """Number of examples in dataset."""
        return len(self.examples)

    def __len__(self) -> int:
        return len(self.examples)

    def __iter__(self):
        return iter(self.examples)

    def __getitem__(self, idx: int) -> QwenExample:
        return self.examples[idx]


class QwenDatasetFromFolder:
    """
    Create QwenExampleSpecs from a folder of images.

    Expects either:
    - Images with matching .txt files (same name)
    - Images with captions in a metadata.json file
    """

    @staticmethod
    def from_folder(
        folder: Path | str,
        default_prompt: str | None = None,
    ) -> list[QwenExampleSpec]:
        """
        Create example specs from a folder.

        Args:
            folder: Path to folder containing images
            default_prompt: Default prompt if no caption found

        Returns:
            List of QwenExampleSpec objects
        """
        folder = Path(folder)
        if not folder.exists():
            raise FileNotFoundError(f"Dataset folder not found: {folder}")

        specs = []
        image_extensions = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

        for image_path in sorted(folder.iterdir()):
            if image_path.suffix.lower() not in image_extensions:
                continue

            # Look for matching caption file
            caption_path = image_path.with_suffix(".txt")
            if caption_path.exists():
                prompt = caption_path.read_text().strip()
            elif default_prompt:
                prompt = default_prompt
            else:
                raise ValueError(f"No caption found for {image_path} and no default_prompt provided")

            specs.append(QwenExampleSpec(prompt=prompt, image=image_path))

        if not specs:
            raise ValueError(f"No images found in {folder}")

        return specs
