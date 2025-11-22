"""
Base classes for declarative weight mapping.

Similar to LoRA mapping, but for transforming HuggingFace weights to MLX structure.
"""

from dataclasses import dataclass
from typing import Callable, List, Optional, Protocol

import mlx.core as mx


@dataclass
class WeightTarget:
    """
    Declarative weight mapping target.

    Maps HuggingFace weight names to MLX nested structure paths.
    """

    mlx_path: str  # MLX nested path, e.g., "transformer_blocks.{block}.attn.to_q.weight"
    hf_patterns: List[str]  # HuggingFace naming patterns, e.g., ["transformer_blocks.{block}.attn.to_q.weight"]
    transform: Optional[Callable[[mx.array], mx.array]] = None  # Optional transform (reshape, transpose, etc.)
    required: bool = True  # If False, weight is optional (may not exist in all models)
    max_blocks: Optional[int] = None  # Override num_blocks for this target (e.g., for fixed-size lists)


class WeightMapping(Protocol):
    """Protocol for weight mapping classes."""

    @staticmethod
    def get_mapping() -> List[WeightTarget]:
        """Return list of weight mapping targets."""
        return []
