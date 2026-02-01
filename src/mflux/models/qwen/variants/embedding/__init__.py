"""Native MLX Qwen3-VL Embedding and Reranker models.

This module provides native MLX implementations of the Qwen3-VL-Embedding-2B
and Qwen3-VL-Reranker-2B models for Apple Silicon inference.

Target: 5-10x speedup over PyTorch with quality parity.

Usage:
    from mflux.models.qwen.variants.embedding import (
        Qwen3VLEmbedder,
        Qwen3VLReranker,
    )

    # Embedding model
    embedder = Qwen3VLEmbedder.from_pretrained("Qwen/Qwen3-VL-Embedding-2B")
    embeddings = embedder.process([{"text": "hello", "instruction": "..."}])

    # Reranker model
    reranker = Qwen3VLReranker.from_pretrained("Qwen/Qwen3-VL-Reranker-2B")
    scores = reranker.process({"query": {"text": "..."}, "documents": [...]})
"""

from .qwen3_vl_embedding import Qwen3VLEmbedder, Qwen3VLEmbeddingConfig
from .qwen3_vl_reranker import Qwen3VLReranker

__all__ = [
    "Qwen3VLEmbedder",
    "Qwen3VLEmbeddingConfig",
    "Qwen3VLReranker",
]
