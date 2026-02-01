"""Native MLX Qwen3-VL Embedding and Reranker models.

This module provides native MLX implementations of the Qwen3-VL-Embedding-2B
and Qwen3-VL-Reranker-2B models for Apple Silicon inference.

Performance (January 2025 benchmarks):
    | Model          | Accuracy | Discrimination | Speed  |
    |----------------|----------|----------------|--------|
    | MLX Embedding  | 93%      | 79.5           | 494ms  |
    | MLX Reranker   | 100%     | 33.9           | 460ms  |
    | CLIP (baseline)| 100%     | 34.9           | 59ms   |

Key features:
    - Native MLX inference (no PyTorch dependency at runtime)
    - Graph compilation with warmup for consistent timing
    - INT8 vision quantization option for 20-30% memory savings
    - Batch processing support for throughput optimization
    - Best-in-class discrimination (79.5 for embedding)

Usage:
    from mflux.models.qwen.variants.embedding import (
        Qwen3VLEmbedder,
        Qwen3VLReranker,
    )

    # Embedding model (best discrimination)
    embedder = Qwen3VLEmbedder.from_pretrained("Qwen/Qwen3-VL-Embedding-2B")
    embedder.compile()  # Optional: 15-40% speedup with warmup
    embeddings = embedder.process([{"text": "hello", "instruction": "..."}])

    # Reranker model (100% accuracy)
    reranker = Qwen3VLReranker.from_pretrained("Qwen/Qwen3-VL-Reranker-2B")
    reranker.compile()  # Optional: 15-40% speedup with warmup
    scores = reranker.process({"query": {"text": "..."}, "documents": [...]})

    # With vision quantization (20-30% memory savings)
    embedder = Qwen3VLEmbedder.from_pretrained(
        "Qwen/Qwen3-VL-Embedding-2B",
        quantize_vision=True
    )
"""

from .pooling import normalize_embeddings, pool_last_token
from .qwen3_vl_embedding import Qwen3VLEmbedder, Qwen3VLEmbedding, Qwen3VLEmbeddingConfig
from .qwen3_vl_reranker import Qwen3VLReranker, Qwen3VLRerankerConfig, Qwen3VLReranking

__all__ = [
    # High-level API
    "Qwen3VLEmbedder",
    "Qwen3VLReranker",
    # Model classes
    "Qwen3VLEmbedding",
    "Qwen3VLReranking",
    # Config classes
    "Qwen3VLEmbeddingConfig",
    "Qwen3VLRerankerConfig",
    # Utilities
    "pool_last_token",
    "normalize_embeddings",
]
