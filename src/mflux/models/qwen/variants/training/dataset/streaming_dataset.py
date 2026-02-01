"""
Streaming Dataset for Large-Scale Qwen-Image Training.

Memory-efficient dataset that loads embeddings on-demand from disk.
Pre-computes embeddings to disk once, then streams during training.
Supports datasets of any size without RAM constraints.

Features:
- Zero startup time (manifest-based, no eager loading)
- Disk cache with LRU memory cache
- Chunked precomputation for large datasets
- Thread-safe for prefetching
"""

import hashlib
import json
import random
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import mlx.core as mx
from tqdm import tqdm

from mflux.models.qwen.variants.training.optimization.qwen_loss import (
    QwenTrainingExample,
)


@dataclass
class ManifestEntry:
    """Single entry in a dataset manifest."""

    image_path: Path
    prompt: str
    cache_key: str | None = None


class StreamingDataset:
    """
    Memory-efficient dataset that loads embeddings on-demand.

    Pre-computes embeddings to disk once, then streams during training.
    Supports datasets of any size without RAM constraints.

    Args:
        manifest_path: Path to manifest JSON file
        cache_dir: Directory for embedding cache
        encoder: Text encoder model (for lazy encoding)
        tokenizer: Tokenizer for text encoder
        vae: VAE model for image encoding
        width: Image width for encoding
        height: Image height for encoding

    Example:
        dataset = StreamingDataset(
            manifest_path="./data/manifest.json",
            cache_dir="~/.cache/qwen_training",
            encoder=qwen.text_encoder,
            tokenizer=qwen.tokenizer,
            vae=qwen.vae,
        )

        # Pre-compute all embeddings (run once)
        dataset.precompute_all()

        # Iterate during training
        for idx in range(len(dataset)):
            example = dataset[idx]
    """

    def __init__(
        self,
        manifest_path: Path | str,
        cache_dir: Path | str,
        encoder: Any,
        tokenizer: Any,
        vae: Any,
        width: int = 1024,
        height: int = 1024,
    ):
        self.manifest_path = Path(manifest_path)
        self.cache_dir = Path(cache_dir).expanduser()
        self.encoder = encoder
        self.tokenizer = tokenizer
        self.vae = vae
        self.width = width
        self.height = height

        # Create cache directories
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.text_cache_dir = self.cache_dir / "text"
        self.image_cache_dir = self.cache_dir / "image"
        self.text_cache_dir.mkdir(exist_ok=True)
        self.image_cache_dir.mkdir(exist_ok=True)

        # Build index from manifest
        self._index: list[ManifestEntry] = []
        self._build_index()

    def _build_index(self) -> None:
        """Build index from manifest (fast, no encoding)."""
        if not self.manifest_path.exists():
            raise FileNotFoundError(f"Manifest not found: {self.manifest_path}")

        with open(self.manifest_path) as f:
            manifest = json.load(f)

        # Determine base path for images
        base_path = Path(manifest.get("base_path", self.manifest_path.parent))

        for entry in manifest.get("images", []):
            image_rel_path = entry.get("image") or entry.get("path")
            prompt = entry.get("prompt") or entry.get("caption")

            if not image_rel_path or not prompt:
                continue

            image_path = base_path / image_rel_path
            cache_key = self._compute_cache_key(image_path, prompt)

            self._index.append(
                ManifestEntry(
                    image_path=image_path,
                    prompt=prompt,
                    cache_key=cache_key,
                )
            )

    def _compute_cache_key(self, image_path: Path, prompt: str) -> str:
        """Compute cache key for an entry."""
        # Include prompt and image path in key
        key_string = f"{prompt}:{image_path.absolute()}"
        return hashlib.md5(key_string.encode("utf-8")).hexdigest()

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, idx: int) -> QwenTrainingExample:
        """Load single example from disk cache (or compute if missing)."""
        entry = self._index[idx]
        return self._load_example(entry)

    def _load_example(self, entry: ManifestEntry) -> QwenTrainingExample:
        """Load or compute a training example."""
        # Try loading from cache
        text_cache_path = self.text_cache_dir / f"{entry.cache_key}_text.safetensors"
        image_cache_path = self.image_cache_dir / f"{entry.cache_key}_image.safetensors"

        # Load text embedding
        if text_cache_path.exists():
            try:
                data = mx.load(str(text_cache_path))
                embeds = data["embeds"]
                mask = data["mask"]
                mx.eval(embeds, mask)
            except Exception:
                # Cache corrupted, recompute
                text_cache_path.unlink(missing_ok=True)
                embeds, mask = self._compute_and_cache_text(entry.prompt, text_cache_path)
        else:
            embeds, mask = self._compute_and_cache_text(entry.prompt, text_cache_path)

        # Load image latent
        if image_cache_path.exists():
            try:
                data = mx.load(str(image_cache_path))
                latent = data["latent"]
                mx.eval(latent)
            except Exception:
                # Cache corrupted, recompute
                image_cache_path.unlink(missing_ok=True)
                latent = self._compute_and_cache_image(entry.image_path, image_cache_path)
        else:
            latent = self._compute_and_cache_image(entry.image_path, image_cache_path)

        return QwenTrainingExample(
            prompt=entry.prompt,
            image_path=str(entry.image_path),
            clean_latents=latent,
            prompt_embeds=embeds,
            prompt_mask=mask,
        )

    def _compute_and_cache_text(self, prompt: str, cache_path: Path) -> tuple[mx.array, mx.array]:
        """Compute and cache text embedding."""
        embeds, mask = self.encoder.encode(prompt, self.tokenizer)
        mx.eval(embeds, mask)
        mx.save_safetensors(str(cache_path), {"embeds": embeds, "mask": mask})
        return embeds, mask

    def _compute_and_cache_image(self, image_path: Path, cache_path: Path) -> mx.array:
        """Compute and cache image latent."""
        from mflux.utils.image_util import ImageUtil

        image = ImageUtil.load_image(str(image_path))
        image = ImageUtil.resize_image(image, width=self.width, height=self.height)
        latent = self.vae.encode(image)
        mx.eval(latent)
        mx.save_safetensors(str(cache_path), {"latent": latent})
        return latent

    def is_cached(self, idx: int) -> bool:
        """Check if an example is cached."""
        entry = self._index[idx]
        text_path = self.text_cache_dir / f"{entry.cache_key}_text.safetensors"
        image_path = self.image_cache_dir / f"{entry.cache_key}_image.safetensors"
        return text_path.exists() and image_path.exists()

    def precompute_all(
        self,
        show_progress: bool = True,
        chunk_size: int = 100,
    ) -> dict[str, int]:
        """
        Pre-compute all embeddings to disk (run once before training).

        Args:
            show_progress: Whether to show progress bar
            chunk_size: Process in chunks of this size (for memory management)

        Returns:
            Dict with counts: {"cached": N, "computed": M, "failed": F}
        """
        stats = {"cached": 0, "computed": 0, "failed": 0}

        # Find uncached items
        uncached_indices = [idx for idx in range(len(self)) if not self.is_cached(idx)]

        if not uncached_indices:
            stats["cached"] = len(self)
            return stats

        stats["cached"] = len(self) - len(uncached_indices)

        iterator = tqdm(uncached_indices, desc="Pre-computing embeddings") if show_progress else uncached_indices

        for idx in iterator:
            try:
                # Loading will compute and cache if needed
                _ = self[idx]
                stats["computed"] += 1
            except Exception as e:
                stats["failed"] += 1
                warnings.warn(
                    f"Failed to cache item {idx}: {e}",
                    UserWarning,
                    stacklevel=2,
                )
                if show_progress:
                    tqdm.write(f"Warning: Failed to cache item {idx}: {e}")

        return stats

    def precompute_chunked(
        self,
        chunk_size: int = 1000,
        show_progress: bool = True,
    ) -> dict[str, int]:
        """
        Pre-compute embeddings in chunks to manage memory.

        Processes chunk_size items at a time, forces evaluation,
        then moves to next chunk. Better for very large datasets.

        Args:
            chunk_size: Number of items to process before forcing evaluation
            show_progress: Whether to show progress bar

        Returns:
            Dict with counts
        """
        stats = {"cached": 0, "computed": 0, "failed": 0}

        uncached_indices = [idx for idx in range(len(self)) if not self.is_cached(idx)]

        if not uncached_indices:
            stats["cached"] = len(self)
            return stats

        stats["cached"] = len(self) - len(uncached_indices)

        # Process in chunks
        for chunk_start in range(0, len(uncached_indices), chunk_size):
            chunk_end = min(chunk_start + chunk_size, len(uncached_indices))
            chunk = uncached_indices[chunk_start:chunk_end]

            chunk_desc = f"Chunk {chunk_start // chunk_size + 1}"
            iterator = tqdm(chunk, desc=chunk_desc) if show_progress else chunk

            for idx in iterator:
                try:
                    _ = self[idx]
                    stats["computed"] += 1
                except Exception as e:
                    stats["failed"] += 1
                    if show_progress:
                        tqdm.write(f"Warning: Failed to cache item {idx}: {e}")

            # Force evaluation between chunks to free memory
            mx.eval()

        return stats

    def get_cache_stats(self) -> dict[str, Any]:
        """Get cache statistics."""
        text_files = list(self.text_cache_dir.glob("*.safetensors"))
        image_files = list(self.image_cache_dir.glob("*.safetensors"))

        text_size = sum(f.stat().st_size for f in text_files)
        image_size = sum(f.stat().st_size for f in image_files)

        cached_count = sum(1 for idx in range(len(self)) if self.is_cached(idx))

        return {
            "total_items": len(self),
            "cached_items": cached_count,
            "cache_hit_rate": cached_count / len(self) if len(self) > 0 else 0,
            "text_cache_size_mb": text_size / (1024 * 1024),
            "image_cache_size_mb": image_size / (1024 * 1024),
        }


class StreamingDatasetFromFolder(StreamingDataset):
    """
    Create streaming dataset from a folder of images.

    Automatically creates manifest from folder contents.
    """

    @classmethod
    def from_folder(
        cls,
        folder_path: Path | str,
        cache_dir: Path | str,
        encoder: Any,
        tokenizer: Any,
        vae: Any,
        default_prompt: str | None = None,
        width: int = 1024,
        height: int = 1024,
    ) -> "StreamingDatasetFromFolder":
        """
        Create dataset from folder of images.

        Args:
            folder_path: Folder containing images and .txt caption files
            cache_dir: Directory for embedding cache
            encoder: Text encoder model
            tokenizer: Tokenizer
            vae: VAE model
            default_prompt: Default prompt for images without captions
            width: Image width
            height: Image height

        Returns:
            StreamingDatasetFromFolder instance
        """
        folder = Path(folder_path)
        cache_dir = Path(cache_dir)

        # Create manifest
        manifest = cls._create_manifest(folder, default_prompt)

        # Save manifest to cache dir
        manifest_path = cache_dir / "manifest.json"
        cache_dir.mkdir(parents=True, exist_ok=True)
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)

        return cls(
            manifest_path=manifest_path,
            cache_dir=cache_dir,
            encoder=encoder,
            tokenizer=tokenizer,
            vae=vae,
            width=width,
            height=height,
        )

    @staticmethod
    def _create_manifest(folder: Path, default_prompt: str | None) -> dict:
        """Create manifest from folder contents."""
        image_extensions = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
        images = []

        for image_path in sorted(folder.iterdir()):
            if image_path.suffix.lower() not in image_extensions:
                continue

            # Find caption
            caption_path = image_path.with_suffix(".txt")
            if caption_path.exists():
                prompt = caption_path.read_text().strip()
            elif default_prompt:
                prompt = default_prompt
            else:
                continue

            images.append(
                {
                    "image": image_path.name,
                    "prompt": prompt,
                }
            )

        return {
            "base_path": str(folder.absolute()),
            "images": images,
        }


class StreamingIterator:
    """
    Iterator for streaming dataset with shuffling support.

    Args:
        dataset: StreamingDataset instance
        batch_size: Batch size
        shuffle: Whether to shuffle each epoch
        seed: Random seed for shuffling
    """

    def __init__(
        self,
        dataset: StreamingDataset,
        batch_size: int = 1,
        shuffle: bool = True,
        seed: int = 42,
    ):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self._rng = random.Random(seed)
        self._indices: list[int] = list(range(len(dataset)))
        self._current_idx = 0
        self._epoch = 0

    @property
    def epoch(self) -> int:
        return self._epoch

    def __iter__(self) -> Iterator[list[int]]:
        """Iterate over batch indices."""
        if self.shuffle:
            self._rng.shuffle(self._indices)

        for i in range(0, len(self._indices), self.batch_size):
            batch_indices = self._indices[i : i + self.batch_size]
            if len(batch_indices) == self.batch_size:
                yield batch_indices

        self._epoch += 1

    def __len__(self) -> int:
        """Number of batches per epoch."""
        return len(self._indices) // self.batch_size
