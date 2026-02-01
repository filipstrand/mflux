"""
Training Validator for Qwen-Image Training.

Provides validation during training:
- Held-out validation loss computation
- Sample image generation at intervals
- Progress tracking and logging

Features:
- Separate validation dataset support
- Configurable validation frequency
- Image generation with EMA weights (if available)
"""

import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import mlx.core as mx

from mflux.models.common.config.config import Config
from mflux.models.qwen.variants.training.optimization.qwen_loss import (
    QwenLoss,
    QwenTrainingBatch,
)


@dataclass
class ValidationResult:
    """Result of a validation run."""

    step: int
    loss: float
    generated_images: list[Path] = field(default_factory=list)


class TrainingValidator:
    """
    Validation during training.

    Features:
    - Held-out validation loss computation
    - Sample image generation at intervals
    - Works with streaming datasets

    Args:
        val_dataset: Validation dataset (StreamingDataset or list of examples)
        val_prompts: Prompts for sample image generation
        output_dir: Directory to save validation images
        val_loss_every: Compute validation loss every N optimizer steps
        generate_every: Generate sample images every N optimizer steps
        num_val_batches: Number of batches for validation loss (default: 10)
        val_batch_size: Batch size for validation (default: 4)

    Example:
        validator = TrainingValidator(
            val_dataset=val_data,
            val_prompts=["a dragon", "a castle"],
            output_dir="./output/validation",
            val_loss_every=100,
            generate_every=500,
        )

        for step, batch in enumerate(dataloader):
            loss, grads = train_step(batch)
            optimizer.update(model, grads)

            if validator.should_validate(step):
                val_result = validator.validate(model, config, step)
                print(f"Step {step}: val_loss={val_result.loss:.4f}")

            if validator.should_generate(step):
                validator.generate_samples(model, step)
    """

    def __init__(
        self,
        val_dataset: Any | None = None,
        val_prompts: list[str] | None = None,
        output_dir: Path | str = "./validation_output",
        val_loss_every: int = 100,
        generate_every: int = 500,
        num_val_batches: int = 10,
        val_batch_size: int = 4,
        seed: int = 42,
    ):
        self.val_dataset = val_dataset
        self.val_prompts = val_prompts or []
        self.output_dir = Path(output_dir).expanduser()
        self.val_loss_every = val_loss_every
        self.generate_every = generate_every
        self.num_val_batches = num_val_batches
        self.val_batch_size = val_batch_size
        self.seed = seed

        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Track validation history
        self._val_losses: list[tuple[int, float]] = []
        self._rng = random.Random(seed)

    @property
    def val_losses(self) -> list[tuple[int, float]]:
        """List of (step, loss) pairs."""
        return list(self._val_losses)

    @property
    def best_val_loss(self) -> float | None:
        """Best (lowest) validation loss seen."""
        if not self._val_losses:
            return None
        return min(loss for _, loss in self._val_losses)

    def should_validate(self, step: int) -> bool:
        """Check if validation loss should be computed at this step."""
        if self.val_dataset is None:
            return False
        return step > 0 and step % self.val_loss_every == 0

    def should_generate(self, step: int) -> bool:
        """Check if sample images should be generated at this step."""
        if not self.val_prompts:
            return False
        return step > 0 and step % self.generate_every == 0

    def validate(
        self,
        model: Any,
        config: Config,
        step: int,
        ema: Any | None = None,
    ) -> ValidationResult:
        """
        Compute validation loss.

        Args:
            model: QwenImage model
            config: Training configuration
            step: Current training step
            ema: Optional EMA model (will use EMA weights if provided)

        Returns:
            ValidationResult with loss and generated images
        """
        if self.val_dataset is None:
            return ValidationResult(step=step, loss=0.0)

        # Apply EMA weights if available
        if ema is not None:
            ema.apply_shadow(model)

        try:
            losses = []

            for _ in range(self.num_val_batches):
                batch = self._sample_val_batch()
                if batch is None:
                    break

                loss = QwenLoss.compute_loss(model, config, batch)
                mx.eval(loss)
                losses.append(float(loss.item()))

            avg_loss = sum(losses) / len(losses) if losses else 0.0

        finally:
            # Restore original weights if using EMA
            if ema is not None:
                ema.restore(model)

        # Track history
        self._val_losses.append((step, avg_loss))

        return ValidationResult(step=step, loss=avg_loss)

    def _sample_val_batch(self) -> QwenTrainingBatch | None:
        """Sample a batch from validation dataset."""
        if self.val_dataset is None:
            return None

        # Handle different dataset types
        if hasattr(self.val_dataset, "__len__") and hasattr(self.val_dataset, "__getitem__"):
            # Indexable dataset
            dataset_len = len(self.val_dataset)
            if dataset_len == 0:
                return None

            indices = self._rng.sample(
                range(dataset_len),
                min(self.val_batch_size, dataset_len),
            )
            examples = [self.val_dataset[idx] for idx in indices]
        elif isinstance(self.val_dataset, list):
            # List of examples
            if not self.val_dataset:
                return None
            examples = self._rng.sample(
                self.val_dataset,
                min(self.val_batch_size, len(self.val_dataset)),
            )
        else:
            return None

        rng = random.Random(self._rng.randint(0, 2**32 - 1))
        return QwenTrainingBatch(examples=examples, rng=rng)

    def generate_samples(
        self,
        model: Any,
        step: int,
        ema: Any | None = None,
        num_inference_steps: int = 20,
    ) -> list[Path]:
        """
        Generate sample images for visual inspection.

        Args:
            model: QwenImage model
            step: Current training step
            ema: Optional EMA model (will use EMA weights if provided)
            num_inference_steps: Number of inference steps for generation

        Returns:
            List of paths to generated images
        """
        if not self.val_prompts:
            return []

        # Apply EMA weights if available
        if ema is not None:
            ema.apply_shadow(model)

        generated_paths = []

        try:
            for i, prompt in enumerate(self.val_prompts):
                try:
                    # Generate image
                    image = model.generate_image(
                        prompt=prompt,
                        seed=self.seed + i,
                        num_inference_steps=num_inference_steps,
                    )

                    # Save image
                    image_path = self.output_dir / f"step_{step:07d}_sample_{i}.png"
                    image.save(str(image_path))
                    generated_paths.append(image_path)

                    # Clean up
                    mx.eval(image)
                    del image

                except Exception as e:
                    print(f"Warning: Failed to generate sample {i}: {e}")

        finally:
            # Restore original weights if using EMA
            if ema is not None:
                ema.restore(model)

            # Clear any cached state
            if hasattr(model, "prompt_cache"):
                model.prompt_cache = {}

        return generated_paths

    def get_summary(self) -> dict[str, Any]:
        """Get validation summary statistics."""
        if not self._val_losses:
            return {
                "num_validations": 0,
                "best_loss": None,
                "last_loss": None,
                "improvement": None,
            }

        best_loss = self.best_val_loss
        last_loss = self._val_losses[-1][1]
        first_loss = self._val_losses[0][1]

        return {
            "num_validations": len(self._val_losses),
            "best_loss": best_loss,
            "last_loss": last_loss,
            "first_loss": first_loss,
            "improvement": (first_loss - last_loss) / first_loss if first_loss > 0 else 0,
            "val_history": self._val_losses,
        }


class NoOpValidator:
    """
    No-op validator for when validation is disabled.

    Provides the same interface but does nothing.
    """

    def should_validate(self, step: int) -> bool:
        return False

    def should_generate(self, step: int) -> bool:
        return False

    def validate(self, model, config, step, ema=None) -> ValidationResult:
        return ValidationResult(step=step, loss=0.0)

    def generate_samples(self, model, step, ema=None, num_inference_steps=20) -> list[Path]:
        return []

    def get_summary(self) -> dict[str, Any]:
        return {"num_validations": 0}
