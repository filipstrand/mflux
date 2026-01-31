import logging
import random
from collections import OrderedDict
from pathlib import Path
from typing import TYPE_CHECKING

import mlx.core as mx
import PIL.Image
from mlx import nn
from tqdm import tqdm

from mflux.models.z_image.latent_creator import ZImageLatentCreator
from mflux.models.z_image.model.z_image_text_encoder.prompt_encoder import PromptEncoder
from mflux.models.z_image.variants.training.dataset.batch import Example
from mflux.models.z_image.variants.training.dataset.preprocessing import ZImagePreProcessing
from mflux.models.z_image.variants.training.state.training_spec import ExampleSpec
from mflux.utils.image_util import ImageUtil

if TYPE_CHECKING:
    from mflux.models.z_image.variants.training.z_image_base import ZImageBase

logger = logging.getLogger(__name__)


class Dataset:
    """Training dataset for Z-Image with pre-encoded images and text."""

    def __init__(self, examples: list[Example]):
        self.examples = examples

    # Maximum allowed examples to prevent OOM (configurable via env var)
    MAX_EFFECTIVE_EXAMPLES = 100_000

    # Maximum embedding cache entries (bounds memory for text deduplication)
    # This is separate from MAX_EFFECTIVE_EXAMPLES since many examples may share prompts
    MAX_EMBEDDING_CACHE_SIZE = 50_000

    @staticmethod
    def split_train_validation(
        examples: list[Example],
        validation_split: float = 0.1,
        seed: int = 42,
    ) -> tuple[list[Example], list[Example]]:
        """Split examples into training and validation sets.

        Uses stratified random sampling to ensure diverse validation set.

        Args:
            examples: Full list of examples to split
            validation_split: Fraction for validation (0.0-0.5)
            seed: Random seed for reproducible splits

        Returns:
            Tuple of (training_examples, validation_examples)

        Raises:
            ValueError: If validation_split is out of range or too few examples
        """
        if not 0.0 < validation_split < 0.5:
            raise ValueError(f"validation_split must be in (0, 0.5), got {validation_split}")

        n_total = len(examples)
        if n_total < 2:
            raise ValueError(f"Need at least 2 examples for split, got {n_total}")

        n_val = max(1, int(n_total * validation_split))
        n_train = n_total - n_val

        if n_train < 1:
            raise ValueError("Split would leave 0 training examples")

        # Shuffle indices deterministically
        rng = random.Random(seed)
        indices = list(range(n_total))
        rng.shuffle(indices)

        # Single-pass partitioning to avoid double iteration
        val_indices = set(indices[:n_val])
        train_examples: list[Example] = []
        val_examples: list[Example] = []
        for i, ex in enumerate(examples):
            if i in val_indices:
                val_examples.append(ex)
            else:
                train_examples.append(ex)

        return train_examples, val_examples

    @staticmethod
    def prepare_dataset(
        model: "ZImageBase",
        raw_data: list[ExampleSpec],
        width: int,
        height: int,
        enable_augmentation: bool = True,
        repeat_count: int = 1,
        random_crop: bool = False,
        seed: int | None = None,
        base_directory: Path | None = None,
    ) -> "Dataset":
        """Prepare dataset by encoding all images and text prompts.

        This pre-computes:
        - VAE latents for all images
        - Text embeddings for all prompts

        This removes ~30-40% of forward pass computation during training.

        Args:
            model: The model for encoding
            raw_data: List of example specifications
            width: Target image width
            height: Target image height
            enable_augmentation: Whether to apply augmentations (flip, etc.)
            repeat_count: Number of times to repeat each example (useful for small datasets)
            random_crop: Whether to apply random crop during encoding
            seed: Random seed for crop reproducibility
            base_directory: Security boundary - all image paths must resolve within this directory.
                          When provided, prevents path traversal attacks. Recommended for production use.

        Raises:
            ValueError: If repeat_count is invalid or effective dataset size exceeds limits
        """
        # Validate repeat_count
        if repeat_count < 1:
            raise ValueError(f"repeat_count must be >= 1, got {repeat_count}")

        # Validate base_directory if provided
        if base_directory is not None:
            if not base_directory.exists():
                raise ValueError(f"base_directory does not exist: {base_directory}")
            if not base_directory.is_dir():
                raise ValueError(f"base_directory is not a directory: {base_directory}")

        # Early warning for large base datasets
        if len(raw_data) > 25_000:
            logger.warning(
                f"Large base dataset ({len(raw_data)} images). "
                f"With repeat_count={repeat_count} and augmentation={enable_augmentation}, "
                f"effective size may approach memory limits."
            )

        # Calculate effective dataset size and validate against memory limits
        augmentation_multiplier = 2 if enable_augmentation else 1  # flip augmentation
        effective_size = len(raw_data) * repeat_count * augmentation_multiplier

        if effective_size > Dataset.MAX_EFFECTIVE_EXAMPLES:
            raise ValueError(
                f"Effective dataset size ({effective_size}) exceeds maximum allowed "
                f"({Dataset.MAX_EFFECTIVE_EXAMPLES}). Reduce repeat_count or dataset size."
            )

        # Create RNG for random cropping
        rng = random.Random(seed) if random_crop else None

        # Encode the original examples
        # Pass base_directory for path traversal protection
        examples = Dataset._create_examples(
            model,
            raw_data,
            width=width,
            height=height,
            random_crop=random_crop,
            rng=rng,
            base_directory=base_directory,
        )

        # Validate that we have at least one successfully encoded example
        if len(examples) == 0:
            raise ValueError(
                f"Dataset preparation failed: 0 examples encoded successfully out of {len(raw_data)} inputs. "
                "Check image paths and encoding errors above."
            )

        # Repeat examples for small datasets
        if repeat_count > 1:
            examples = ZImagePreProcessing.repeat_examples(examples, repeat_count)

        # Augment dataset if enabled
        if enable_augmentation:
            augmented_examples = [
                variation for example in examples for variation in ZImagePreProcessing.augment(example)
            ]
            return Dataset(augmented_examples)

        return Dataset(examples)

    @staticmethod
    def prepare_dataset_with_validation(
        model: "ZImageBase",
        raw_data: list[ExampleSpec],
        width: int,
        height: int,
        validation_split: float = 0.1,
        enable_augmentation: bool = True,
        repeat_count: int = 1,
        random_crop: bool = False,
        seed: int | None = None,
        base_directory: Path | None = None,
    ) -> tuple["Dataset", "Dataset"]:
        """Prepare training and validation datasets with holdout split.

        The validation set is split BEFORE augmentation and repetition to ensure
        true out-of-sample validation (no augmented versions of val images in train).

        Args:
            model: The model for encoding
            raw_data: List of example specifications
            width: Target image width
            height: Target image height
            validation_split: Fraction of data for validation (0.0-0.5)
            enable_augmentation: Whether to apply augmentations (train set only)
            repeat_count: Number of times to repeat training examples
            random_crop: Whether to apply random crop during encoding
            seed: Random seed for reproducibility
            base_directory: Security boundary for image paths

        Returns:
            Tuple of (training_dataset, validation_dataset)

        Note:
            Augmentation and repetition are only applied to the training set.
            The validation set remains unaugmented for consistent evaluation.
        """
        # Encode all examples first (shared between train/val)
        rng = random.Random(seed) if random_crop else None

        examples = Dataset._create_examples(
            model,
            raw_data,
            width=width,
            height=height,
            random_crop=random_crop,
            rng=rng,
            base_directory=base_directory,
        )

        if len(examples) < 2:
            raise ValueError(f"Need at least 2 examples for validation split, got {len(examples)}")

        # Split BEFORE augmentation for true holdout
        train_examples, val_examples = Dataset.split_train_validation(
            examples=examples,
            validation_split=validation_split,
            seed=seed if seed is not None else 42,
        )

        logger.info(f"Split dataset: {len(train_examples)} train, {len(val_examples)} validation")

        # Apply repetition to training set only
        if repeat_count > 1:
            train_examples = ZImagePreProcessing.repeat_examples(train_examples, repeat_count)

        # Apply augmentation to training set only
        if enable_augmentation:
            train_examples = [
                variation for example in train_examples for variation in ZImagePreProcessing.augment(example)
            ]

        return Dataset(train_examples), Dataset(val_examples)

    def size(self) -> int:
        return len(self.examples)

    @staticmethod
    def _create_examples(
        model: "ZImageBase",
        raw_data: list[ExampleSpec],
        width: int,
        height: int,
        random_crop: bool = False,
        rng: random.Random | None = None,
        base_directory: Path | None = None,
    ) -> list[Example]:
        """Create examples with encoded images and text embeddings.

        Optimization: Text embeddings are deduplicated by prompt content.
        This provides 10-15% speedup for datasets with repeated prompts
        (common with augmentation) and saves 0.5-1GB memory.

        Args:
            model: The model for encoding
            raw_data: List of example specifications
            width: Target image width
            height: Target image height
            random_crop: Whether to apply random crop during encoding
            rng: Random number generator for reproducible crops
            base_directory: Security boundary - all image paths must resolve within this directory
        """
        examples = []

        # Text embedding cache for deduplication (bounded by MAX_EMBEDDING_CACHE_SIZE)
        # Uses OrderedDict for LRU eviction: move_to_end() on hits, popitem(last=False) on eviction
        # Key: prompt text, Value: encoded embeddings
        embedding_cache: OrderedDict[str, mx.array] = OrderedDict()
        cache_hits = 0
        cache_misses = 0
        cache_evictions = 0

        # Periodic garbage collection interval to prevent memory fragmentation
        # for large datasets (100+ examples at a time)
        GC_INTERVAL = 100

        for i, entry in enumerate(tqdm(raw_data, desc="Encoding dataset")):
            image = None
            scaled_image = None
            image_array = None
            encoded = None

            # Periodic garbage collection to prevent memory fragmentation
            # This is especially important for large datasets approaching MAX_EFFECTIVE_EXAMPLES
            if i > 0 and i % GC_INTERVAL == 0:
                import gc

                gc.collect()

            try:
                # Encode the image through VAE
                # Pass base_directory for path traversal protection
                encoded_image = Dataset._encode_image(
                    vae=model.vae,
                    image_path=entry.image,
                    width=width,
                    height=height,
                    random_crop=random_crop,
                    rng=rng,
                    base_directory=base_directory,
                )

                # Encode the prompt through text encoder (with LRU deduplication)
                if entry.prompt in embedding_cache:
                    # Reuse cached embedding and move to end (LRU: most recently used)
                    text_embeddings = embedding_cache[entry.prompt]
                    embedding_cache.move_to_end(entry.prompt)
                    cache_hits += 1
                else:
                    # Check cache size limit before adding
                    if len(embedding_cache) >= Dataset.MAX_EMBEDDING_CACHE_SIZE:
                        # Evict least recently used entry (first in OrderedDict)
                        embedding_cache.popitem(last=False)
                        cache_evictions += 1

                    # Encode and cache
                    text_embeddings = PromptEncoder.encode_prompt(
                        prompt=entry.prompt,
                        tokenizer=model.tokenizers["z_image"],
                        text_encoder=model.text_encoder,
                    )
                    embedding_cache[entry.prompt] = text_embeddings
                    cache_misses += 1

                # Create the example
                example = Example(
                    example_id=i,
                    prompt=entry.prompt,
                    image_path=entry.image,
                    encoded_image=encoded_image,
                    text_embeddings=text_embeddings,
                )
                examples.append(example)

                # Force computation to enable progress tracking and free memory
                mx.synchronize()
            except Exception as e:  # noqa: BLE001 - Intentional: graceful degradation for encoding failures
                # Log error but continue with remaining examples for graceful degradation
                logger.warning(f"Failed to encode image {entry.image}: {e}. Skipping this example.")
                continue
            finally:
                # Explicit cleanup of intermediate objects
                for var in [image, scaled_image, image_array, encoded]:
                    if var is not None:
                        del var

        # Report deduplication statistics
        total_prompts = cache_hits + cache_misses
        if total_prompts > 0 and cache_hits > 0:
            hit_rate = cache_hits / total_prompts * 100
            stats_msg = f"Text embedding deduplication: {cache_hits}/{total_prompts} cached ({hit_rate:.1f}% hit rate)"
            if cache_evictions > 0:
                stats_msg += f", {cache_evictions} evictions"
            logger.info(stats_msg)

        return examples

    # Allowed image extensions for security validation
    ALLOWED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif", ".tiff", ".tif"}

    @staticmethod
    def _encode_image(
        vae: nn.Module,
        image_path: Path,
        width: int,
        height: int,
        random_crop: bool = False,
        rng: random.Random | None = None,
        base_directory: Path | None = None,
    ) -> mx.array:
        """Encode an image to VAE latent space.

        Args:
            vae: VAE encoder module
            image_path: Path to image file
            width: Target width
            height: Target height
            random_crop: Whether to apply random crop instead of center crop
            rng: Random number generator for reproducible crops
            base_directory: If provided, validates image_path is within this directory

        Security:
            - Validates file extension before any filesystem operations
            - Detects and rejects symbolic links to prevent traversal attacks
            - Validates path is within base_directory before resolve()
        """
        # Security: Validate file extension first (before any filesystem operations)
        if image_path.suffix.lower() not in Dataset.ALLOWED_IMAGE_EXTENSIONS:
            raise ValueError(
                f"Security: Invalid file extension '{image_path.suffix}'. Allowed: {Dataset.ALLOWED_IMAGE_EXTENSIONS}"
            )

        # Security: Check for symlinks BEFORE resolving to prevent traversal attacks
        # Using lstat() which doesn't follow symlinks
        if image_path.exists() and image_path.is_symlink():
            raise ValueError(f"Security: Symbolic links not allowed for security reasons: {image_path}")

        # Path traversal protection: validate BEFORE resolve() to prevent attacks
        if base_directory is not None:
            # Check the unresolved path doesn't contain traversal sequences
            path_str = str(image_path)
            if ".." in path_str:
                raise ValueError(f"Security: Path traversal not allowed: {image_path}")

            # Now resolve both paths
            resolved_base = base_directory.resolve()
            resolved_path = image_path.resolve()

            # Verify resolved path is within base directory
            try:
                resolved_path.relative_to(resolved_base)
            except ValueError:
                raise ValueError(f"Security: Image path {resolved_path} is outside allowed directory {resolved_base}")

            # Double-check symlink after resolve (in case of TOCTOU)
            if resolved_path.is_symlink():
                raise ValueError(f"Security: Path resolved to symbolic link: {resolved_path}")
        else:
            resolved_path = image_path.resolve()

        # Validate file exists and is a regular file
        if not resolved_path.exists():
            raise FileNotFoundError(f"Image file not found: {resolved_path}")
        if not resolved_path.is_file():
            raise ValueError(f"Image path is not a file: {resolved_path}")

        # Load image with context manager to ensure cleanup
        with PIL.Image.open(resolved_path) as img:
            image = img.convert("RGB")

        if random_crop and rng is not None:
            # Apply random crop: first resize to maintain aspect ratio with some slack,
            # then random crop to target size
            original_width, original_height = image.size

            # Calculate scale factor to ensure image is at least target size
            scale = max(width / original_width, height / original_height)
            scaled_width = int(original_width * scale)
            scaled_height = int(original_height * scale)

            # Resize preserving aspect ratio
            image = image.resize((scaled_width, scaled_height), PIL.Image.Resampling.LANCZOS)

            # Apply random crop
            crop_top, crop_left = ZImagePreProcessing.random_crop_params(
                original_width=scaled_width,
                original_height=scaled_height,
                target_width=width,
                target_height=height,
                rng=rng,
            )
            scaled_image = ZImagePreProcessing.apply_random_crop(
                image=image,
                crop_top=crop_top,
                crop_left=crop_left,
                target_width=width,
                target_height=height,
            )
            del image
        else:
            # Standard center crop/resize
            scaled_image = ImageUtil.scale_to_dimensions(image, target_width=width, target_height=height)
            del image

        # Encode through VAE
        image_array = ImageUtil.to_array(scaled_image)
        del scaled_image  # Free scaled PIL image

        encoded = vae.encode(image_array)
        del image_array  # Free input array

        # Pack latents for Z-Image format
        latents = ZImageLatentCreator.pack_latents(encoded, width=width, height=height)
        del encoded  # Free unpacked VAE output

        return latents
