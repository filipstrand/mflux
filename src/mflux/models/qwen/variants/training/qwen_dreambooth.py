"""
Qwen-Image DreamBooth Training.

Main training orchestrator for Qwen-Image LoRA/DoRA fine-tuning.
Supports:
- LoRA and DoRA adapters
- Learning rate scheduling
- Gradient accumulation
- EMA model tracking
- Checkpointing and resumption
"""

import mlx.core as mx
from mlx import nn
from tqdm import tqdm

from mflux.models.common.config.config import Config
from mflux.models.common.lora.layer.adapter_factory import AdapterType
from mflux.models.common.lora.layer.dora_linear_layer import DoRALinear
from mflux.models.common.lora.layer.linear_lora_layer import LoRALinear
from mflux.models.qwen.variants.training.optimization.qwen_loss import QwenLoss
from mflux.models.qwen.variants.training.state.qwen_training_spec import QwenTrainingSpec
from mflux.models.qwen.variants.training.state.qwen_training_state import QwenTrainingState


class QwenDreamBooth:
    """
    DreamBooth-style training for Qwen-Image.

    Adapts the DreamBooth training approach for Qwen-Image models
    with support for LoRA, DoRA, and full fine-tuning.
    """

    @staticmethod
    def train(
        qwen,  # QwenImage model
        config: Config,
        spec: QwenTrainingSpec,
        state: QwenTrainingState,
    ) -> None:
        """
        Main training loop.

        Args:
            qwen: QwenImage model with LoRA/DoRA layers applied
            config: Model configuration
            spec: Training specification
            state: Training state (iterator, optimizer, etc.)
        """
        # Freeze base model
        qwen.freeze()

        # Unfreeze adapter layers (LoRA/DoRA)
        adapter_type = AdapterType.from_string(spec.lora_layers.adapter_type)
        QwenDreamBooth._unfreeze_adapter_layers(qwen.transformer, adapter_type)

        # Define loss computation as function of batch
        train_step_fn = nn.value_and_grad(
            model=qwen,
            fn=lambda batch: QwenLoss.compute_loss(qwen, config, batch),
        )

        # Setup progress bar
        total_steps = spec.total_steps
        progress = tqdm(
            state.iterator,
            total=total_steps,
            initial=state.iterator.num_iterations,
            desc="Training",
        )

        # Track optimizer steps for accurate progress
        optimizer_steps = 0

        # Training loop
        for batch in progress:
            try:
                # Forward + backward pass
                loss, grads = train_step_fn(batch)

                # Force evaluation of loss before accumulation to free memory
                loss_value = float(loss.item())
                mx.eval(loss)

                # Gradient accumulation
                accumulated_grads = state.grad_accumulator.accumulate(grads)

                if accumulated_grads is not None:
                    # Apply gradients
                    state.optimizer.optimizer.update(
                        model=qwen,
                        gradients=accumulated_grads,
                    )

                    # Force evaluation after optimizer step
                    mx.eval(qwen.parameters())

                    # Update learning rate (steps once per optimizer update)
                    lr = state.lr_scheduler.step()
                    optimizer_steps += 1

                    # Update EMA if enabled
                    if state.ema is not None:
                        state.ema.update(qwen)

                # Track loss
                state.statistics.add_loss(
                    iteration=state.iterator.num_iterations,
                    loss=loss_value,
                )

                # Update progress bar
                progress.set_postfix(
                    {
                        "loss": f"{loss_value:.4f}",
                        "lr": f"{state.lr_scheduler.get_lr():.2e}",
                        "epoch": state.iterator.epoch,
                        "opt_steps": optimizer_steps,
                    }
                )

                # Clean up to free memory
                del loss, grads
                if accumulated_grads is not None:
                    del accumulated_grads

                # Validation image generation
                if state.should_validate(spec):
                    QwenDreamBooth._generate_validation_image(
                        qwen=qwen,
                        spec=spec,
                        state=state,
                    )

                # Save checkpoint
                if state.should_save(spec):
                    checkpoint_path = state.save_checkpoint(qwen, spec)
                    tqdm.write(f"Saved checkpoint: {checkpoint_path}")

            except KeyboardInterrupt:
                tqdm.write("\nTraining interrupted. Saving checkpoint...")
                state.save_checkpoint(qwen, spec)
                raise

        # Handle any remaining accumulated gradients (partial window at end of training)
        if state.grad_accumulator.is_accumulating:
            tqdm.write(
                f"Applying partial accumulated gradients ({state.grad_accumulator.count}/{state.grad_accumulator.accumulation_steps} steps)"
            )
            partial_grads = state.grad_accumulator.flush()
            if partial_grads is not None:
                state.optimizer.optimizer.update(
                    model=qwen,
                    gradients=partial_grads,
                )
                mx.eval(qwen.parameters())
                state.lr_scheduler.step()

                # Update EMA with final weights (important for model quality)
                if state.ema is not None:
                    state.ema.update(qwen)

        # Final checkpoint
        tqdm.write("Training complete. Saving final checkpoint...")
        state.save_checkpoint(qwen, spec)

    @staticmethod
    def _unfreeze_adapter_layers(
        module: nn.Module,
        adapter_type: AdapterType,
    ) -> None:
        """
        Unfreeze LoRA/DoRA layers for training.

        Args:
            module: Model module to search
            adapter_type: Type of adapter to unfreeze
        """
        for name, child in module.named_modules():
            if isinstance(child, LoRALinear):
                child.unfreeze(keys=["lora_A", "lora_B"], strict=False)
            elif isinstance(child, DoRALinear):
                child.unfreeze(keys=["lora_A", "lora_B", "magnitude"], strict=False)

    @staticmethod
    def _generate_validation_image(
        qwen,
        spec: QwenTrainingSpec,
        state: QwenTrainingState,
    ) -> None:
        """Generate validation image during training."""
        if spec.instrumentation is None:
            return

        prompt = spec.instrumentation.validation_prompt
        if not prompt:
            return

        # Use EMA weights for validation if available
        if state.ema is not None:
            state.ema.apply_shadow(qwen)

        try:
            image = qwen.generate_image(
                seed=spec.seed,
                prompt=prompt,
                num_inference_steps=spec.steps,
                height=spec.height,
                width=spec.width,
                guidance=spec.guidance,
            )

            # Save validation image
            output_dir = f"{spec.saver.output_path}/_validation/images/"
            import os

            os.makedirs(output_dir, exist_ok=True)
            image_path = f"{output_dir}/{state.iterator.num_iterations:07d}_validation.png"
            image.save(path=image_path)
            tqdm.write(f"Saved validation image: {image_path}")

            # Force evaluation of image computation graph before deletion
            # to ensure proper memory cleanup
            mx.eval(image)
            del image
            qwen.prompt_cache = {}  # Clear prompt cache

        finally:
            # Restore original weights if using EMA
            if state.ema is not None:
                state.ema.restore(qwen)


class QwenFullFinetune:
    """
    Full fine-tuning for Qwen-Image (no LoRA).

    For systems with sufficient memory (128GB+ unified).
    """

    @staticmethod
    def train(
        qwen,
        config: Config,
        spec: QwenTrainingSpec,
        state: QwenTrainingState,
        freeze_text_encoder: bool = True,
        freeze_vae: bool = True,
    ) -> None:
        """
        Full fine-tuning training loop.

        Args:
            qwen: QwenImage model
            config: Model configuration
            spec: Training specification
            state: Training state
            freeze_text_encoder: Whether to freeze text encoder
            freeze_vae: Whether to freeze VAE
        """
        # Optionally freeze components
        if freeze_text_encoder:
            qwen.text_encoder.freeze()
        if freeze_vae:
            qwen.vae.freeze()

        # Leave transformer unfrozen for full fine-tuning
        # Note: This requires significant memory!

        train_step_fn = nn.value_and_grad(
            model=qwen,
            fn=lambda batch: QwenLoss.compute_loss(qwen, config, batch),
        )

        progress = tqdm(
            state.iterator,
            total=spec.total_steps,
            initial=state.iterator.num_iterations,
            desc="Full Fine-tuning",
        )

        # Track optimizer steps for accurate progress
        optimizer_steps = 0

        for batch in progress:
            try:
                loss, grads = train_step_fn(batch)

                # Force evaluation of loss before accumulation to free memory
                loss_value = float(loss.item())
                mx.eval(loss)

                accumulated_grads = state.grad_accumulator.accumulate(grads)

                if accumulated_grads is not None:
                    state.optimizer.optimizer.update(
                        model=qwen,
                        gradients=accumulated_grads,
                    )

                    # Force evaluation after optimizer step
                    mx.eval(qwen.parameters())

                    state.lr_scheduler.step()
                    optimizer_steps += 1

                    if state.ema is not None:
                        state.ema.update(qwen)

                state.statistics.add_loss(
                    iteration=state.iterator.num_iterations,
                    loss=loss_value,
                )

                progress.set_postfix(
                    {
                        "loss": f"{loss_value:.4f}",
                        "lr": f"{state.lr_scheduler.get_lr():.2e}",
                        "opt_steps": optimizer_steps,
                    }
                )

                # Clean up to free memory
                del loss, grads
                if accumulated_grads is not None:
                    del accumulated_grads

                if state.should_save(spec):
                    checkpoint_path = state.save_checkpoint(qwen, spec)
                    tqdm.write(f"Saved checkpoint: {checkpoint_path}")

            except KeyboardInterrupt:
                tqdm.write("\nTraining interrupted. Saving checkpoint...")
                state.save_checkpoint(qwen, spec)
                raise

        # Handle any remaining accumulated gradients
        if state.grad_accumulator.is_accumulating:
            tqdm.write(
                f"Applying partial accumulated gradients ({state.grad_accumulator.count}/{state.grad_accumulator.accumulation_steps} steps)"
            )
            partial_grads = state.grad_accumulator.flush()
            if partial_grads is not None:
                state.optimizer.optimizer.update(
                    model=qwen,
                    gradients=partial_grads,
                )
                mx.eval(qwen.parameters())
                state.lr_scheduler.step()

                # Update EMA with final weights (important for model quality)
                if state.ema is not None:
                    state.ema.update(qwen)

        tqdm.write("Training complete. Saving final checkpoint...")
        state.save_checkpoint(qwen, spec)
