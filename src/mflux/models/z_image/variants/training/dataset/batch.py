import random
from dataclasses import dataclass
from pathlib import Path

import mlx.core as mx


@dataclass
class Example:
    """A single training example with encoded image and text embeddings."""

    example_id: int
    prompt: str
    image_path: Path
    encoded_image: mx.array  # VAE-encoded latent
    text_embeddings: mx.array  # Text encoder output (cap_feats)

    # Optional augmentation tracking
    augmentation_id: int = 0
    is_flipped: bool = False


@dataclass
class Batch:
    """A batch of training examples.

    When aspect ratio bucketing is enabled, all examples in the batch
    share the same target resolution (from the bucket assignment).
    """

    examples: list[Example]
    rng: random.Random
    target_resolution: tuple[int, int] | None = None  # (width, height) for aspect ratio bucketing

    def __len__(self) -> int:
        return len(self.examples)

    @property
    def batch_size(self) -> int:
        return len(self.examples)

    @property
    def target_width(self) -> int | None:
        """Target width when using aspect ratio bucketing."""
        return self.target_resolution[0] if self.target_resolution else None

    @property
    def target_height(self) -> int | None:
        """Target height when using aspect ratio bucketing."""
        return self.target_resolution[1] if self.target_resolution else None
