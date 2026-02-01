"""
Embedding Cache for Qwen-Image Training.

Pre-computes and caches text and image embeddings to avoid
redundant encoder passes during training. This provides
2-3x speedup for typical training runs.

Features:
- Caches text embeddings (prompt -> embedding)
- Caches image latents (image path -> VAE latent)
- Uses file modification time for cache invalidation
- Supports batch precomputation before training
- Proper LRU eviction for memory cache using OrderedDict
- Configurable memory budget for large-scale training
"""

import hashlib
from collections import OrderedDict
from pathlib import Path
from typing import Any, Callable

import mlx.core as mx
from tqdm import tqdm


class EmbeddingCache:
    """
    Cache for pre-computed embeddings and latents.

    Stores text prompt embeddings and image VAE latents to disk,
    avoiding redundant encoder passes during training.

    Uses OrderedDict for proper LRU (Least Recently Used) eviction
    in the in-memory cache.

    Args:
        cache_dir: Directory to store cached embeddings
        max_memory_items: Maximum items in memory cache (default: 100)

    Example:
        cache = EmbeddingCache(Path("~/.cache/qwen_training"))

        # Pre-compute all embeddings before training
        cache.precompute_all(dataset, text_encoder, tokenizer, vae)

        # During training, use cached embeddings
        for example in dataset:
            embeds, mask = cache.get_text_embedding(
                example.prompt,
                encoder=text_encoder,
                tokenizer=tokenizer,
            )
            latent = cache.get_image_latent(
                example.image_path,
                vae=vae,
            )
    """

    def __init__(self, cache_dir: Path | str, max_memory_items: int = 100):
        self.cache_dir = Path(cache_dir).expanduser()
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Subdirectories for organization
        self.text_cache_dir = self.cache_dir / "text"
        self.image_cache_dir = self.cache_dir / "image"
        self.text_cache_dir.mkdir(exist_ok=True)
        self.image_cache_dir.mkdir(exist_ok=True)

        # In-memory LRU caches using OrderedDict for proper LRU eviction
        # OrderedDict maintains insertion order; move_to_end() on access makes it LRU
        self._text_memory_cache: OrderedDict[str, tuple[mx.array, mx.array]] = OrderedDict()
        self._image_memory_cache: OrderedDict[str, mx.array] = OrderedDict()
        self._max_memory_items = max_memory_items

    def _get_text_cache_key(self, prompt: str) -> str:
        """Generate cache key for text prompt."""
        return hashlib.md5(prompt.encode("utf-8")).hexdigest()

    def _get_image_cache_key(self, image_path: Path | str, mtime: float | None = None) -> str:
        """
        Generate cache key for image.

        Includes file modification time for cache invalidation.
        """
        path = Path(image_path)
        if mtime is None:
            mtime = path.stat().st_mtime

        key_string = f"{path.absolute()}:{mtime}"
        return hashlib.md5(key_string.encode("utf-8")).hexdigest()

    def get_text_embedding(
        self,
        prompt: str,
        encoder: Any,
        tokenizer: Any,
        compute_fn: Callable[[str, Any, Any], tuple[mx.array, mx.array]] | None = None,
    ) -> tuple[mx.array, mx.array]:
        """
        Get text embedding, computing and caching if necessary.

        Args:
            prompt: Text prompt to encode
            encoder: Text encoder model
            tokenizer: Tokenizer for text encoder
            compute_fn: Optional custom compute function.
                Default uses encoder.encode(prompt, tokenizer)

        Returns:
            Tuple of (embeddings, attention_mask)
        """
        cache_key = self._get_text_cache_key(prompt)

        # Check memory cache first (move to end for LRU on hit)
        if cache_key in self._text_memory_cache:
            self._text_memory_cache.move_to_end(cache_key)
            return self._text_memory_cache[cache_key]

        # Check disk cache
        cache_path = self.text_cache_dir / f"{cache_key}.safetensors"
        if cache_path.exists():
            try:
                data = mx.load(str(cache_path))
                embeds = data["embeds"]
                mask = data["mask"]
                # Force evaluation after loading from disk to prevent lazy graph issues
                mx.eval(embeds, mask)
                self._add_to_text_memory_cache(cache_key, (embeds, mask))
                return embeds, mask
            except Exception:
                # Cache file corrupted, delete and recompute
                cache_path.unlink(missing_ok=True)

        # Compute embedding
        try:
            if compute_fn is not None:
                embeds, mask = compute_fn(prompt, encoder, tokenizer)
            else:
                # Default: assume encoder has encode method
                embeds, mask = encoder.encode(prompt, tokenizer)
        except Exception as e:
            raise RuntimeError(f"Failed to encode text prompt: {e}") from e

        # Force evaluation
        mx.eval(embeds, mask)

        # Save to disk cache
        mx.save_safetensors(str(cache_path), {"embeds": embeds, "mask": mask})

        # Add to memory cache
        self._add_to_text_memory_cache(cache_key, (embeds, mask))

        return embeds, mask

    def get_image_latent(
        self,
        image_path: Path | str,
        vae: Any,
        compute_fn: Callable[[Path, Any], mx.array] | None = None,
    ) -> mx.array:
        """
        Get image latent, computing and caching if necessary.

        Args:
            image_path: Path to image file
            vae: VAE model for encoding
            compute_fn: Optional custom compute function.
                Default uses vae.encode(load_image(path))

        Returns:
            Image latent tensor
        """
        path = Path(image_path)

        # Get file modification time for cache key
        try:
            mtime = path.stat().st_mtime
        except FileNotFoundError:
            raise FileNotFoundError(f"Image not found: {path}")

        cache_key = self._get_image_cache_key(path, mtime)

        # Check memory cache first (move to end for LRU on hit)
        if cache_key in self._image_memory_cache:
            self._image_memory_cache.move_to_end(cache_key)
            return self._image_memory_cache[cache_key]

        # Check disk cache
        cache_path = self.image_cache_dir / f"{cache_key}.safetensors"
        if cache_path.exists():
            try:
                data = mx.load(str(cache_path))
                latent = data["latent"]
                # Force evaluation after loading from disk to prevent lazy graph issues
                mx.eval(latent)
                self._add_to_image_memory_cache(cache_key, latent)
                return latent
            except Exception:
                # Cache file corrupted, delete and recompute
                cache_path.unlink(missing_ok=True)

        # Compute latent
        try:
            if compute_fn is not None:
                latent = compute_fn(path, vae)
            else:
                # Default: use vae.encode with image loading
                from mflux.utils.image_util import ImageUtil

                image = ImageUtil.load_image(str(path))
                latent = vae.encode(image)
        except Exception as e:
            raise RuntimeError(f"Failed to encode image {path}: {e}") from e

        # Force evaluation
        mx.eval(latent)

        # Save to disk cache
        mx.save_safetensors(str(cache_path), {"latent": latent})

        # Add to memory cache
        self._add_to_image_memory_cache(cache_key, latent)

        return latent

    def _add_to_text_memory_cache(self, key: str, value: tuple[mx.array, mx.array]) -> None:
        """Add to text memory cache with proper LRU eviction."""
        # If key already exists, move to end (most recently used)
        if key in self._text_memory_cache:
            self._text_memory_cache.move_to_end(key)
            self._text_memory_cache[key] = value
            return

        # Evict least recently used (first item) if at capacity
        if len(self._text_memory_cache) >= self._max_memory_items:
            # popitem(last=False) removes the first (oldest/LRU) item
            self._text_memory_cache.popitem(last=False)

        # Add new item at end (most recently used)
        self._text_memory_cache[key] = value

    def _add_to_image_memory_cache(self, key: str, value: mx.array) -> None:
        """Add to image memory cache with proper LRU eviction."""
        # If key already exists, move to end (most recently used)
        if key in self._image_memory_cache:
            self._image_memory_cache.move_to_end(key)
            self._image_memory_cache[key] = value
            return

        # Evict least recently used (first item) if at capacity
        if len(self._image_memory_cache) >= self._max_memory_items:
            # popitem(last=False) removes the first (oldest/LRU) item
            self._image_memory_cache.popitem(last=False)

        # Add new item at end (most recently used)
        self._image_memory_cache[key] = value

    def precompute_all(
        self,
        dataset: Any,
        encoder: Any,
        tokenizer: Any,
        vae: Any,
        text_compute_fn: Callable[[str, Any, Any], tuple[mx.array, mx.array]] | None = None,
        image_compute_fn: Callable[[Path, Any], mx.array] | None = None,
        show_progress: bool = True,
    ) -> dict[str, int]:
        """
        Pre-compute all embeddings for a dataset.

        Call this before training to cache all embeddings upfront.
        This avoids encoder overhead during training and provides
        2-3x speedup.

        Args:
            dataset: Iterable of examples with .prompt and .image_path attributes
            encoder: Text encoder model
            tokenizer: Tokenizer for text encoder
            vae: VAE model for image encoding
            text_compute_fn: Optional custom text compute function
            image_compute_fn: Optional custom image compute function
            show_progress: Whether to show progress bar

        Returns:
            Dict with counts: {"text_cached": N, "text_computed": M, "image_cached": K, "image_computed": L, "failed": F}
        """
        import warnings

        stats = {
            "text_cached": 0,
            "text_computed": 0,
            "image_cached": 0,
            "image_computed": 0,
            "failed": 0,
        }

        items = list(dataset)
        iterator = tqdm(items, desc="Caching embeddings") if show_progress else items

        for example in iterator:
            # Cache text embedding
            prompt = getattr(example, "prompt", None) or getattr(example, "caption", None)
            if prompt:
                cache_key = self._get_text_cache_key(prompt)
                cache_path = self.text_cache_dir / f"{cache_key}.safetensors"
                if cache_path.exists():
                    stats["text_cached"] += 1
                else:
                    self.get_text_embedding(prompt, encoder, tokenizer, text_compute_fn)
                    stats["text_computed"] += 1

            # Cache image latent
            image_path = getattr(example, "image_path", None) or getattr(example, "image", None)
            if image_path:
                path = Path(image_path)
                if path.exists():
                    try:
                        mtime = path.stat().st_mtime
                        cache_key = self._get_image_cache_key(path, mtime)
                        cache_path = self.image_cache_dir / f"{cache_key}.safetensors"
                        if cache_path.exists():
                            stats["image_cached"] += 1
                        else:
                            self.get_image_latent(path, vae, image_compute_fn)
                            stats["image_computed"] += 1
                    except Exception as e:
                        stats["failed"] += 1
                        # Always warn about failures, regardless of show_progress
                        warnings.warn(f"Failed to cache {path}: {e}", UserWarning, stacklevel=2)
                        if show_progress:
                            tqdm.write(f"Warning: Failed to cache {path}: {e}")

        return stats

    def clear_memory_cache(self) -> None:
        """Clear in-memory caches."""
        self._text_memory_cache.clear()
        self._image_memory_cache.clear()

    def clear_disk_cache(self) -> None:
        """Clear all disk caches."""
        import shutil

        if self.text_cache_dir.exists():
            shutil.rmtree(self.text_cache_dir)
            self.text_cache_dir.mkdir()

        if self.image_cache_dir.exists():
            shutil.rmtree(self.image_cache_dir)
            self.image_cache_dir.mkdir()

    def get_cache_stats(self) -> dict[str, Any]:
        """Get cache statistics."""
        text_files = list(self.text_cache_dir.glob("*.safetensors"))
        image_files = list(self.image_cache_dir.glob("*.safetensors"))

        text_size = sum(f.stat().st_size for f in text_files)
        image_size = sum(f.stat().st_size for f in image_files)

        return {
            "text_cached_count": len(text_files),
            "image_cached_count": len(image_files),
            "text_cache_size_mb": text_size / (1024 * 1024),
            "image_cache_size_mb": image_size / (1024 * 1024),
            "text_memory_cache_count": len(self._text_memory_cache),
            "image_memory_cache_count": len(self._image_memory_cache),
        }


def _estimate_array_bytes(arr: mx.array) -> int:
    """Estimate memory usage of an MLX array in bytes."""
    # Get dtype size
    dtype_sizes = {
        mx.float32: 4,
        mx.float16: 2,
        mx.bfloat16: 2,
        mx.int32: 4,
        mx.int64: 8,
        mx.uint32: 4,
        mx.bool_: 1,
    }
    dtype_size = dtype_sizes.get(arr.dtype, 4)  # Default to 4 bytes
    return arr.size * dtype_size


class MemoryBudgetCache(EmbeddingCache):
    """
    Embedding cache with configurable memory budget.

    Extends EmbeddingCache with memory-budget-based eviction instead of
    item-count-based eviction. Better for large-scale training where
    you want to maximize cache utilization within memory constraints.

    Args:
        cache_dir: Directory to store cached embeddings
        memory_budget_gb: Maximum memory to use for in-memory cache (default: 10.0)
        min_items: Minimum items to keep regardless of memory (default: 10)

    Example:
        # Use 20GB for in-memory cache
        cache = MemoryBudgetCache(
            cache_dir=Path("~/.cache/qwen_training"),
            memory_budget_gb=20.0,
        )

        # Pre-compute all embeddings
        cache.precompute_all(dataset, encoder, tokenizer, vae)

        # During training, cache will hold as many items as fit in 20GB
        for batch in dataloader:
            ...
    """

    def __init__(
        self,
        cache_dir: Path | str,
        memory_budget_gb: float = 10.0,
        min_items: int = 10,
    ):
        # Initialize with very large max_items - we'll manage by memory instead
        super().__init__(cache_dir=cache_dir, max_memory_items=1_000_000)

        self._memory_budget_bytes = int(memory_budget_gb * 1024**3)
        self._min_items = min_items

        # Track memory usage per item
        self._text_item_sizes: dict[str, int] = {}
        self._image_item_sizes: dict[str, int] = {}
        self._current_text_memory = 0
        self._current_image_memory = 0

    @property
    def memory_budget_gb(self) -> float:
        """Memory budget in gigabytes."""
        return self._memory_budget_bytes / (1024**3)

    @property
    def current_memory_usage_gb(self) -> float:
        """Current memory usage in gigabytes."""
        return (self._current_text_memory + self._current_image_memory) / (1024**3)

    def _add_to_text_memory_cache(self, key: str, value: tuple[mx.array, mx.array]) -> None:
        """Add to text memory cache with memory-budget-based eviction."""
        embeds, mask = value
        item_bytes = _estimate_array_bytes(embeds) + _estimate_array_bytes(mask)

        # If key already exists, update size tracking
        if key in self._text_memory_cache:
            old_size = self._text_item_sizes.get(key, 0)
            self._current_text_memory -= old_size
            self._text_memory_cache.move_to_end(key)
            self._text_memory_cache[key] = value
            self._text_item_sizes[key] = item_bytes
            self._current_text_memory += item_bytes
            return

        # Evict LRU items until we have space (keep at least min_items)
        total_memory = self._current_text_memory + self._current_image_memory
        while total_memory + item_bytes > self._memory_budget_bytes and len(self._text_memory_cache) > self._min_items:
            # Remove oldest item
            evicted_key, _ = self._text_memory_cache.popitem(last=False)
            evicted_size = self._text_item_sizes.pop(evicted_key, 0)
            self._current_text_memory -= evicted_size
            total_memory = self._current_text_memory + self._current_image_memory

        # Add new item
        self._text_memory_cache[key] = value
        self._text_item_sizes[key] = item_bytes
        self._current_text_memory += item_bytes

    def _add_to_image_memory_cache(self, key: str, value: mx.array) -> None:
        """Add to image memory cache with memory-budget-based eviction."""
        item_bytes = _estimate_array_bytes(value)

        # If key already exists, update size tracking
        if key in self._image_memory_cache:
            old_size = self._image_item_sizes.get(key, 0)
            self._current_image_memory -= old_size
            self._image_memory_cache.move_to_end(key)
            self._image_memory_cache[key] = value
            self._image_item_sizes[key] = item_bytes
            self._current_image_memory += item_bytes
            return

        # Evict LRU items until we have space (keep at least min_items)
        total_memory = self._current_text_memory + self._current_image_memory
        while total_memory + item_bytes > self._memory_budget_bytes and len(self._image_memory_cache) > self._min_items:
            # Remove oldest item
            evicted_key, _ = self._image_memory_cache.popitem(last=False)
            evicted_size = self._image_item_sizes.pop(evicted_key, 0)
            self._current_image_memory -= evicted_size
            total_memory = self._current_text_memory + self._current_image_memory

        # Add new item
        self._image_memory_cache[key] = value
        self._image_item_sizes[key] = item_bytes
        self._current_image_memory += item_bytes

    def clear_memory_cache(self) -> None:
        """Clear in-memory caches and reset memory tracking."""
        super().clear_memory_cache()
        self._text_item_sizes.clear()
        self._image_item_sizes.clear()
        self._current_text_memory = 0
        self._current_image_memory = 0

    def get_cache_stats(self) -> dict[str, Any]:
        """Get cache statistics including memory usage."""
        stats = super().get_cache_stats()
        stats.update(
            {
                "memory_budget_gb": self.memory_budget_gb,
                "current_memory_usage_gb": self.current_memory_usage_gb,
                "text_memory_usage_mb": self._current_text_memory / (1024 * 1024),
                "image_memory_usage_mb": self._current_image_memory / (1024 * 1024),
            }
        )
        return stats


def create_cache(
    cache_dir: Path | str,
    memory_budget_gb: float | None = None,
    max_items: int | None = None,
) -> EmbeddingCache:
    """
    Factory function to create appropriate cache type.

    Args:
        cache_dir: Directory to store cached embeddings
        memory_budget_gb: If provided, create MemoryBudgetCache with this budget
        max_items: If provided (and memory_budget_gb is None), create EmbeddingCache
            with this item limit. Default: 100

    Returns:
        EmbeddingCache or MemoryBudgetCache instance

    Example:
        # Item-count-based cache (default, small datasets)
        cache = create_cache("~/.cache/qwen", max_items=100)

        # Memory-budget-based cache (large datasets)
        cache = create_cache("~/.cache/qwen", memory_budget_gb=20.0)
    """
    if memory_budget_gb is not None:
        return MemoryBudgetCache(
            cache_dir=cache_dir,
            memory_budget_gb=memory_budget_gb,
        )
    else:
        return EmbeddingCache(
            cache_dir=cache_dir,
            max_memory_items=max_items or 100,
        )
