"""
Qwen-Image Training Batch and Example Classes.

Data structures for organizing training examples and batches
for Qwen-Image LoRA/DoRA/full fine-tuning.
"""

from pathlib import Path
from random import Random

import mlx.core as mx


class QwenExample:
    """
    Single training example for Qwen-Image.

    Stores pre-encoded latents and embeddings for efficient training.

    Attributes:
        example_id: Unique identifier for this example
        prompt: Original text prompt
        image_name: Original image path (for reference)
        clean_latents: VAE-encoded image latents
        prompt_embeds: Text encoder output embeddings
        prompt_mask: Attention mask for text encoder output
    """

    def __init__(
        self,
        example_id: int,
        prompt: str,
        image_path: str | Path,
        encoded_image: mx.array,
        prompt_embeds: mx.array,
        prompt_mask: mx.array,
    ):
        self.example_id = example_id
        self.prompt = prompt
        self.image_name = str(image_path)
        self.clean_latents = encoded_image
        self.prompt_embeds = prompt_embeds
        self.prompt_mask = prompt_mask

    def __repr__(self) -> str:
        return (
            f"QwenExample(id={self.example_id}, prompt={self.prompt[:30]}..., latent_shape={self.clean_latents.shape})"
        )


class QwenBatch:
    """
    Batch of training examples for Qwen-Image.

    Provides utilities for stacking examples into batched tensors.

    Attributes:
        examples: List of QwenExample objects in this batch
        rng: Random number generator for this batch (for reproducibility)
    """

    def __init__(self, examples: list[QwenExample], rng: Random):
        self.examples = examples
        self.rng = rng

    @property
    def batch_size(self) -> int:
        return len(self.examples)

    @property
    def max_seq_len(self) -> int:
        """Maximum sequence length in this batch."""
        return max(ex.prompt_embeds.shape[1] for ex in self.examples)

    def get_stacked_clean_latents(self) -> mx.array:
        """
        Stack clean latents into batched tensor.

        Returns:
            Tensor of shape (batch_size, seq_len, hidden_dim)
        """
        # All latents should have same shape for same image dimensions
        return mx.stack([ex.clean_latents for ex in self.examples], axis=0)

    def get_stacked_prompt_embeds(self) -> mx.array:
        """
        Stack prompt embeddings with padding to max sequence length.

        Returns:
            Tensor of shape (batch_size, max_seq_len, hidden_dim)
        """
        max_len = self.max_seq_len
        padded = []
        for ex in self.examples:
            curr_len = ex.prompt_embeds.shape[1]
            if curr_len < max_len:
                pad_len = max_len - curr_len
                # Pad along sequence dimension (axis=1)
                padded_embed = mx.pad(ex.prompt_embeds, [(0, 0), (0, pad_len), (0, 0)])
            else:
                padded_embed = ex.prompt_embeds
            padded.append(padded_embed)
        return mx.concatenate(padded, axis=0)

    def get_stacked_prompt_masks(self) -> mx.array:
        """
        Stack prompt masks with padding (pad with 0 = ignore).

        Returns:
            Tensor of shape (batch_size, max_seq_len)
        """
        max_len = self.max_seq_len
        padded = []
        for ex in self.examples:
            curr_len = ex.prompt_mask.shape[1]
            if curr_len < max_len:
                pad_len = max_len - curr_len
                # Pad with 0 (masked/ignored tokens)
                padded_mask = mx.pad(ex.prompt_mask, [(0, 0), (0, pad_len)])
            else:
                padded_mask = ex.prompt_mask
            padded.append(padded_mask)
        return mx.concatenate(padded, axis=0)

    def get_example_ids(self) -> list[int]:
        """Get list of example IDs in this batch."""
        return [ex.example_id for ex in self.examples]

    def __len__(self) -> int:
        return self.batch_size

    def __iter__(self):
        return iter(self.examples)

    def __repr__(self) -> str:
        return f"QwenBatch(size={self.batch_size}, max_seq_len={self.max_seq_len})"
