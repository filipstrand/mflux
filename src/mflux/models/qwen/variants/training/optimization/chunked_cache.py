"""
Chunked Embedding Cache for Large-Scale Training.

Stores embeddings in chunks rather than individual files for better
disk I/O efficiency. Ideal for datasets with 5k+ images.

Features:
- Chunk-based storage (1000 items per chunk by default)
- LRU eviction at chunk level
- Configurable memory budget
- Better disk I/O patterns for large datasets
"""

import json
from collections import OrderedDict
from pathlib import Path
from typing import Any

import mlx.core as mx
from tqdm import tqdm


def _estimate_array_bytes(arr: mx.array) -> int:
    """Estimate memory usage of an MLX array in bytes."""
    dtype_sizes = {
        mx.float32: 4,
        mx.float16: 2,
        mx.bfloat16: 2,
        mx.int32: 4,
        mx.int64: 8,
        mx.uint32: 4,
        mx.bool_: 1,
    }
    dtype_size = dtype_sizes.get(arr.dtype, 4)
    return arr.size * dtype_size


class ChunkedEmbeddingCache:
    """
    Chunked embedding cache optimized for large datasets.

    Stores embeddings in chunks (e.g., 1000 items per chunk file)
    rather than individual files. This provides:
    - Better disk I/O (fewer files, larger sequential reads)
    - More efficient LRU eviction (chunk-level granularity)
    - Configurable memory budget

    Args:
        cache_dir: Directory to store cache chunks
        memory_budget_gb: Maximum memory for in-memory cache (default: 10.0)
        chunk_size: Number of items per chunk (default: 1000)

    Example:
        cache = ChunkedEmbeddingCache(
            cache_dir="~/.cache/qwen_training",
            memory_budget_gb=20.0,
            chunk_size=1000,
        )

        # Get a batch of items
        examples = cache.get_batch(indices=[0, 1, 2, 3])
    """

    def __init__(
        self,
        cache_dir: Path | str,
        memory_budget_gb: float = 10.0,
        chunk_size: int = 1000,
    ):
        self.cache_dir = Path(cache_dir).expanduser()
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self._memory_budget_bytes = int(memory_budget_gb * 1024**3)
        self._chunk_size = chunk_size

        # Chunk-level LRU cache
        # key: chunk_id, value: dict of {local_idx: (latent, embeds, mask)}
        self._loaded_chunks: OrderedDict[int, dict[int, tuple]] = OrderedDict()
        self._chunk_sizes: dict[int, int] = {}
        self._current_memory = 0

        # Metadata
        self._metadata_path = self.cache_dir / "metadata.json"
        self._metadata: dict[str, Any] = self._load_metadata()

    @property
    def chunk_size(self) -> int:
        return self._chunk_size

    @property
    def memory_budget_gb(self) -> float:
        return self._memory_budget_bytes / (1024**3)

    @property
    def current_memory_usage_gb(self) -> float:
        return self._current_memory / (1024**3)

    def _load_metadata(self) -> dict[str, Any]:
        """Load cache metadata."""
        if self._metadata_path.exists():
            with open(self._metadata_path) as f:
                return json.load(f)
        return {"chunk_size": self._chunk_size, "items": {}}

    def _save_metadata(self) -> None:
        """Save cache metadata."""
        with open(self._metadata_path, "w") as f:
            json.dump(self._metadata, f, indent=2)

    def _get_chunk_id(self, idx: int) -> int:
        """Map item index to chunk ID."""
        return idx // self._chunk_size

    def _get_local_idx(self, idx: int) -> int:
        """Map item index to local index within chunk."""
        return idx % self._chunk_size

    def _get_chunk_path(self, chunk_id: int) -> Path:
        """Get path to chunk file."""
        return self.cache_dir / f"chunk_{chunk_id:06d}.safetensors"

    def is_cached(self, idx: int) -> bool:
        """Check if an item is cached."""
        chunk_id = self._get_chunk_id(idx)
        local_idx = self._get_local_idx(idx)

        # Check memory cache
        if chunk_id in self._loaded_chunks:
            return local_idx in self._loaded_chunks[chunk_id]

        # Check disk
        chunk_path = self._get_chunk_path(chunk_id)
        if not chunk_path.exists():
            return False

        # Check metadata
        item_key = str(idx)
        return item_key in self._metadata.get("items", {})

    def get_item(
        self,
        idx: int,
        compute_fn: Any | None = None,
    ) -> tuple[mx.array, mx.array, mx.array]:
        """
        Get a single item from cache.

        Args:
            idx: Item index
            compute_fn: Optional function(idx) -> (latent, embeds, mask)
                Called if item not cached

        Returns:
            Tuple of (latent, prompt_embeds, prompt_mask)
        """
        chunk_id = self._get_chunk_id(idx)
        local_idx = self._get_local_idx(idx)

        # Check memory cache
        if chunk_id in self._loaded_chunks:
            self._loaded_chunks.move_to_end(chunk_id)
            if local_idx in self._loaded_chunks[chunk_id]:
                return self._loaded_chunks[chunk_id][local_idx]

        # Try to load chunk from disk
        chunk_path = self._get_chunk_path(chunk_id)
        if chunk_path.exists():
            self._load_chunk(chunk_id)
            if local_idx in self._loaded_chunks.get(chunk_id, {}):
                return self._loaded_chunks[chunk_id][local_idx]

        # Compute if not cached
        if compute_fn is None:
            raise KeyError(f"Item {idx} not in cache and no compute_fn provided")

        latent, embeds, mask = compute_fn(idx)
        mx.eval(latent, embeds, mask)

        # Add to cache
        self._add_item(idx, latent, embeds, mask)

        return latent, embeds, mask

    def get_batch(
        self,
        indices: list[int],
        compute_fn: Any | None = None,
    ) -> list[tuple[mx.array, mx.array, mx.array]]:
        """
        Get a batch of items from cache.

        Optimized for sequential access - prefetches entire chunks.

        Args:
            indices: List of item indices
            compute_fn: Optional function(idx) -> (latent, embeds, mask)

        Returns:
            List of (latent, prompt_embeds, prompt_mask) tuples
        """
        results = []

        # Group by chunk for efficient loading
        chunks_needed = set(self._get_chunk_id(idx) for idx in indices)

        # Load needed chunks
        for chunk_id in chunks_needed:
            if chunk_id not in self._loaded_chunks:
                chunk_path = self._get_chunk_path(chunk_id)
                if chunk_path.exists():
                    self._load_chunk(chunk_id)
            else:
                self._loaded_chunks.move_to_end(chunk_id)

        # Retrieve items
        for idx in indices:
            results.append(self.get_item(idx, compute_fn))

        return results

    def _load_chunk(self, chunk_id: int) -> None:
        """Load a chunk from disk into memory cache."""
        chunk_path = self._get_chunk_path(chunk_id)

        if not chunk_path.exists():
            return

        try:
            data = mx.load(str(chunk_path))
        except Exception:
            # Corrupted chunk, skip
            return

        # Parse chunk data
        chunk_data = {}
        chunk_bytes = 0

        for key in data.keys():
            if key.startswith("latent_"):
                local_idx = int(key.split("_")[1])
                latent = data[f"latent_{local_idx}"]
                embeds = data.get(f"embeds_{local_idx}")
                mask = data.get(f"mask_{local_idx}")

                if embeds is not None and mask is not None:
                    mx.eval(latent, embeds, mask)
                    chunk_data[local_idx] = (latent, embeds, mask)
                    chunk_bytes += (
                        _estimate_array_bytes(latent) + _estimate_array_bytes(embeds) + _estimate_array_bytes(mask)
                    )

        if not chunk_data:
            return

        # Evict if needed
        self._evict_if_needed(chunk_bytes)

        # Add to memory cache
        self._loaded_chunks[chunk_id] = chunk_data
        self._chunk_sizes[chunk_id] = chunk_bytes
        self._current_memory += chunk_bytes

    def _add_item(
        self,
        idx: int,
        latent: mx.array,
        embeds: mx.array,
        mask: mx.array,
    ) -> None:
        """Add a single item to cache."""
        chunk_id = self._get_chunk_id(idx)
        local_idx = self._get_local_idx(idx)

        item_bytes = _estimate_array_bytes(latent) + _estimate_array_bytes(embeds) + _estimate_array_bytes(mask)

        # Evict if needed
        self._evict_if_needed(item_bytes)

        # Add to memory cache
        if chunk_id not in self._loaded_chunks:
            self._loaded_chunks[chunk_id] = {}
            self._chunk_sizes[chunk_id] = 0

        self._loaded_chunks[chunk_id][local_idx] = (latent, embeds, mask)
        self._chunk_sizes[chunk_id] += item_bytes
        self._current_memory += item_bytes

        # Update metadata
        self._metadata["items"][str(idx)] = True

    def _evict_if_needed(self, required_bytes: int) -> None:
        """Evict oldest chunks until we have space."""
        while self._current_memory + required_bytes > self._memory_budget_bytes and len(self._loaded_chunks) > 0:
            # Remove oldest chunk (first item in OrderedDict)
            chunk_id, _ = self._loaded_chunks.popitem(last=False)
            evicted_size = self._chunk_sizes.pop(chunk_id, 0)
            self._current_memory -= evicted_size

    def save_chunk(self, chunk_id: int) -> None:
        """Save a chunk from memory to disk."""
        if chunk_id not in self._loaded_chunks:
            return

        chunk_data = self._loaded_chunks[chunk_id]
        if not chunk_data:
            return

        # Build save dict
        save_dict = {}
        for local_idx, (latent, embeds, mask) in chunk_data.items():
            save_dict[f"latent_{local_idx}"] = latent
            save_dict[f"embeds_{local_idx}"] = embeds
            save_dict[f"mask_{local_idx}"] = mask

        chunk_path = self._get_chunk_path(chunk_id)
        mx.save_safetensors(str(chunk_path), save_dict)

    def flush_to_disk(self) -> None:
        """Save all in-memory chunks to disk."""
        for chunk_id in list(self._loaded_chunks.keys()):
            self.save_chunk(chunk_id)
        self._save_metadata()

    def precompute_all(
        self,
        total_items: int,
        compute_fn: Any,
        show_progress: bool = True,
    ) -> dict[str, int]:
        """
        Pre-compute all embeddings in chunks.

        Args:
            total_items: Total number of items to cache
            compute_fn: Function(idx) -> (latent, embeds, mask)
            show_progress: Whether to show progress bar

        Returns:
            Dict with counts: {"cached": N, "computed": M, "failed": F}
        """
        stats = {"cached": 0, "computed": 0, "failed": 0}

        # Find uncached items
        uncached_indices = [idx for idx in range(total_items) if not self.is_cached(idx)]

        if not uncached_indices:
            stats["cached"] = total_items
            return stats

        stats["cached"] = total_items - len(uncached_indices)

        # Group by chunk
        chunks_to_compute: dict[int, list[int]] = {}
        for idx in uncached_indices:
            chunk_id = self._get_chunk_id(idx)
            if chunk_id not in chunks_to_compute:
                chunks_to_compute[chunk_id] = []
            chunks_to_compute[chunk_id].append(idx)

        # Process each chunk
        iterator = (
            tqdm(chunks_to_compute.items(), desc="Pre-computing chunks") if show_progress else chunks_to_compute.items()
        )

        for chunk_id, indices in iterator:
            for idx in indices:
                try:
                    latent, embeds, mask = compute_fn(idx)
                    mx.eval(latent, embeds, mask)
                    self._add_item(idx, latent, embeds, mask)
                    stats["computed"] += 1
                except Exception as e:
                    stats["failed"] += 1
                    if show_progress:
                        tqdm.write(f"Warning: Failed to cache item {idx}: {e}")

            # Save chunk and potentially evict from memory
            self.save_chunk(chunk_id)
            mx.eval()  # Force evaluation between chunks

        self._save_metadata()
        return stats

    def clear_memory_cache(self) -> None:
        """Clear in-memory cache."""
        self._loaded_chunks.clear()
        self._chunk_sizes.clear()
        self._current_memory = 0

    def get_cache_stats(self) -> dict[str, Any]:
        """Get cache statistics."""
        chunk_files = list(self.cache_dir.glob("chunk_*.safetensors"))
        total_size = sum(f.stat().st_size for f in chunk_files)

        return {
            "num_chunks_on_disk": len(chunk_files),
            "num_chunks_in_memory": len(self._loaded_chunks),
            "disk_cache_size_mb": total_size / (1024 * 1024),
            "memory_cache_size_mb": self._current_memory / (1024 * 1024),
            "memory_budget_gb": self.memory_budget_gb,
            "cached_items": len(self._metadata.get("items", {})),
            "chunk_size": self._chunk_size,
        }
