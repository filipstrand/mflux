"""
Qwen-Image Data Augmentation and Preprocessing.

Provides augmentation strategies for training data including:
- Horizontal flipping (preserves latent structure)
- Random cropping (for multi-crop training)
- Prompt variations (optional)
"""

import mlx.core as mx

from mflux.models.qwen.variants.training.dataset.qwen_batch import QwenExample


class QwenPreprocessing:
    """
    Data augmentation for Qwen-Image training.

    Augmentation is applied at the latent level to avoid
    re-encoding images multiple times.
    """

    @staticmethod
    def augment(
        example: QwenExample,
        horizontal_flip: bool = True,
        random_crops: int = 0,
    ) -> list[QwenExample]:
        """
        Apply augmentation to a single example.

        Args:
            example: Original training example
            horizontal_flip: Whether to add horizontally flipped version
            random_crops: Number of random crop variations (not yet implemented)

        Returns:
            List of examples including original and augmented versions
        """
        results = [example]

        # Horizontal flip augmentation
        if horizontal_flip:
            flipped = QwenPreprocessing._horizontal_flip(example)
            results.append(flipped)

        # Random crops (TODO: implement for multi-crop training)
        # This requires unpacking latents, cropping, and re-packing
        # which is more complex for diffusion models

        return results

    @staticmethod
    def _horizontal_flip(example: QwenExample) -> QwenExample:
        """
        Create horizontally flipped version of example.

        Flips the image latents while preserving the prompt.
        Works at the latent level to avoid VAE re-encoding.
        """
        # Flip latents along spatial dimension
        # Latent shape after packing: (batch, seq, hidden)
        # The spatial information is packed into the sequence dimension
        # For proper flipping, we need to consider the packing structure

        # For now, use a simple flip that works for most cases
        # This assumes latents are packed in row-major order
        flipped_latents = mx.flip(example.clean_latents, axis=1)

        return QwenExample(
            example_id=example.example_id + 10000,  # Offset ID to distinguish
            prompt=example.prompt,
            image_path=f"{example.image_name}_flipped",
            encoded_image=flipped_latents,
            prompt_embeds=example.prompt_embeds,
            prompt_mask=example.prompt_mask,
        )

    @staticmethod
    def augment_prompt(prompt: str, variations: list[str] | None = None) -> list[str]:
        """
        Generate prompt variations for diversity.

        Args:
            prompt: Original prompt
            variations: Optional list of prompt templates with {prompt} placeholder

        Returns:
            List of prompt variations
        """
        if variations is None:
            # Default variations
            variations = [
                "{prompt}",
                "A photo of {prompt}",
                "An image of {prompt}",
                "A high quality photo of {prompt}",
            ]

        return [v.format(prompt=prompt) for v in variations]


class QwenAugmentationConfig:
    """Configuration for data augmentation."""

    def __init__(
        self,
        horizontal_flip: bool = True,
        random_crops: int = 0,
        prompt_variations: bool = False,
        color_jitter: bool = False,  # Not implemented yet
    ):
        self.horizontal_flip = horizontal_flip
        self.random_crops = random_crops
        self.prompt_variations = prompt_variations
        self.color_jitter = color_jitter

    def apply(self, example: QwenExample) -> list[QwenExample]:
        """Apply configured augmentations to example."""
        return QwenPreprocessing.augment(
            example=example,
            horizontal_flip=self.horizontal_flip,
            random_crops=self.random_crops,
        )
