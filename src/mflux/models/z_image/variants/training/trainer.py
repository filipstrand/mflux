"""Z-Image trainer for both LoRA and full fine-tuning.

This module provides the main training loop for Z-Image-Base.
It supports:
- LoRA training: Efficient adapter training
- Full fine-tuning: Complete model training (leveraging 512GB memory)
- CFG-aware training
- Checkpoint save/resume
- Validation image generation

Optimizations:
- MLX graph compilation with per-bucket caching (15-40% speedup)
- Vectorized batch loss computation (10-25% speedup)
"""

import logging

import mlx.core as mx
from tqdm import tqdm

from mflux.models.common.config.config import Config
from mflux.models.z_image.variants.training.modes.full_trainer import FullTrainer
from mflux.models.z_image.variants.training.modes.lora_trainer import LoRATrainer
from mflux.models.z_image.variants.training.optimization.compiled_train_step import CompiledTrainStep
from mflux.models.z_image.variants.training.optimization.deferred_sync import create_synchronizer
from mflux.models.z_image.variants.training.optimization.z_image_loss import ZImageLoss
from mflux.models.z_image.variants.training.state.training_spec import TrainingMode, TrainingSpec
from mflux.models.z_image.variants.training.state.training_state import TrainingState
from mflux.models.z_image.variants.training.statistics.plotter import Plotter
from mflux.models.z_image.variants.training.statistics.profiler import create_profiler
from mflux.models.z_image.variants.training.z_image_base import ZImageBase

logger = logging.getLogger(__name__)


class ZImageTrainer:
    """Main trainer for Z-Image-Base.

    Supports both LoRA and full fine-tuning modes.
    """

    @staticmethod
    def train(
        model: ZImageBase,
        config: Config,
        training_spec: TrainingSpec,
        training_state: TrainingState,
    ) -> None:
        """Main training loop.

        Args:
            model: Z-Image-Base model (with LoRA layers if LoRA mode)
            config: Inference config (for validation)
            training_spec: Training specification
            training_state: Training state (iterator, optimizer, statistics)
        """
        # Setup model for training based on mode
        if training_spec.mode == TrainingMode.LORA:
            LoRATrainer.setup_for_training(model)
            LoRATrainer.print_training_info(model)
        else:
            FullTrainer.setup_for_training(model, training_spec.full_finetune)
            FullTrainer.print_training_info(model, training_spec.full_finetune)

        # Define loss function with gradient computation using compiled train step
        # CompiledTrainStep provides:
        # - MLX graph compilation (15-40% speedup)
        # - Per-bucket caching for different aspect ratios
        # - Automatic cache management
        enable_compilation = getattr(training_spec, "enable_compilation", True)
        compiled_train_step = CompiledTrainStep(model, config, enabled=enable_compilation)
        train_step_function = compiled_train_step

        # Setup deferred synchronization (30-40% sync overhead reduction)
        # Sync every 4 updates by default, or per gradient accumulation batch
        sync_interval = getattr(training_spec, "sync_interval", 4)
        deferred_sync = create_synchronizer(sync_interval=sync_interval, adaptive=True)

        # Setup profiler for timing instrumentation
        enable_profiling = getattr(training_spec, "enable_profiling", False)
        profiler = create_profiler(enabled=enable_profiling)

        # Setup progress bar
        batches = tqdm(
            training_state.iterator,
            total=training_state.iterator.total_number_of_steps(),
            initial=training_state.iterator.num_iterations,
            desc="Training",
        )

        # Gradient accumulation and clipping support
        accumulation_steps = training_spec.training_loop.gradient_accumulation_steps
        max_grad_norm = training_spec.optimizer.max_grad_norm
        accumulated_grads = None
        accumulation_count = 0

        # NaN/Inf gradient tracking to detect training divergence
        MAX_CONSECUTIVE_NAN_SKIPS = 10
        consecutive_nan_skips = 0
        total_nan_skips = 0

        # Get EMA and scheduler from training state
        ema = training_state.ema
        scheduler = training_state.scheduler

        # Main training loop
        for batch in batches:
            # Compute loss and gradients
            with profiler.time_section("forward"):
                loss, grads = train_step_function(batch)

            # Gradient accumulation
            if accumulation_steps > 1:
                if accumulated_grads is None:
                    # First step: move grads to accumulated_grads (same reference)
                    accumulated_grads = grads
                else:
                    # Subsequent steps: accumulate and free the new grads
                    old_accumulated_grads = accumulated_grads
                    accumulated_grads = ZImageTrainer._accumulate_grads(accumulated_grads, grads)
                    del old_accumulated_grads  # Free old accumulated dict
                    del grads  # Free immediately after accumulation
                accumulation_count += 1

                if accumulation_count >= accumulation_steps:
                    # Average gradients (scale creates new dict, so free old one)
                    old_accumulated_grads = accumulated_grads
                    accumulated_grads = ZImageTrainer._scale_grads(accumulated_grads, 1.0 / accumulation_steps)
                    del old_accumulated_grads  # Free unscaled gradients
                    # Apply gradient clipping to prevent explosion in 6B parameter model
                    if max_grad_norm > 0:
                        accumulated_grads = ZImageTrainer._clip_grad_norm(accumulated_grads, max_grad_norm)
                        if accumulated_grads is None:
                            # Track consecutive NaN/Inf skips to detect training divergence
                            consecutive_nan_skips += 1
                            total_nan_skips += 1
                            if consecutive_nan_skips >= MAX_CONSECUTIVE_NAN_SKIPS:
                                raise RuntimeError(
                                    f"Training diverged: {MAX_CONSECUTIVE_NAN_SKIPS} consecutive NaN/Inf gradient updates detected. "
                                    f"Total NaN skips: {total_nan_skips}. Consider: (1) reducing learning rate, "
                                    f"(2) enabling gradient clipping, (3) checking dataset for corrupted samples."
                                )
                            accumulation_count = 0
                            continue
                    # Reset consecutive counter on successful update
                    consecutive_nan_skips = 0
                    # Apply update
                    with profiler.time_section("optimizer"):
                        training_state.optimizer.update(model=model, gradients=accumulated_grads)
                    # Update EMA weights after optimizer step
                    if ema is not None:
                        ema.update(model)
                    # Step the learning rate scheduler
                    if scheduler is not None:
                        scheduler.step()
                    with profiler.time_section("sync"):
                        deferred_sync.maybe_sync()  # Deferred sync for better throughput
                    del accumulated_grads  # Free after update
                    accumulated_grads = None
                    accumulation_count = 0
            else:
                # No accumulation, update directly
                # Apply gradient clipping to prevent explosion in 6B parameter model
                if max_grad_norm > 0:
                    grads = ZImageTrainer._clip_grad_norm(grads, max_grad_norm)
                    if grads is None:
                        # Track consecutive NaN/Inf skips to detect training divergence
                        consecutive_nan_skips += 1
                        total_nan_skips += 1
                        if consecutive_nan_skips >= MAX_CONSECUTIVE_NAN_SKIPS:
                            raise RuntimeError(
                                f"Training diverged: {MAX_CONSECUTIVE_NAN_SKIPS} consecutive NaN/Inf gradient updates detected. "
                                f"Total NaN skips: {total_nan_skips}. Consider: (1) reducing learning rate, "
                                f"(2) enabling gradient clipping, (3) checking dataset for corrupted samples."
                            )
                        continue
                # Reset consecutive counter on successful update
                consecutive_nan_skips = 0
                with profiler.time_section("optimizer"):
                    training_state.optimizer.update(model=model, gradients=grads)
                # Update EMA weights after optimizer step
                if ema is not None:
                    ema.update(model)
                # Step the learning rate scheduler
                if scheduler is not None:
                    scheduler.step()
                with profiler.time_section("sync"):
                    deferred_sync.maybe_sync()  # Deferred sync for better throughput
                del grads  # Free after update

            # Update progress bar (float() forces materialization of the loss value)
            loss_value = float(loss)
            batches.set_postfix({"loss": f"{loss_value:.4f}"})

            # Cleanup: delete references to release computation graph memory
            del loss, batch

            # Periodic memory cleanup to prevent fragmentation from lazy evaluation
            # mx.synchronize() is called periodically by deferred_sync

            # Plot loss periodically
            if training_state.should_plot_loss(training_spec):
                deferred_sync.force_sync()  # Ensure all updates complete before validation
                validation_batch = training_state.iterator.get_validation_batch()
                try:
                    with profiler.time_section("validation"):
                        validation_loss = ZImageLoss.compute_validation_loss(model, config, validation_batch)
                    validation_loss_float = float(validation_loss)
                    training_state.statistics.append_values(
                        step=training_state.iterator.num_iterations,
                        loss=validation_loss_float,
                    )
                    Plotter.update_loss_plot(training_spec=training_spec, training_state=training_state)

                    # Early stopping check
                    if training_state.early_stopping is not None:
                        if training_state.early_stopping.check(validation_loss_float):
                            logger.info("Early stopping triggered: saving final checkpoint")
                            training_state.save(model, training_spec)
                            return  # Exit training loop
                finally:
                    del validation_batch
                    if "validation_loss" in locals():
                        del validation_loss

            # Generate validation image periodically
            if training_state.should_generate_image(training_spec):
                deferred_sync.force_sync()  # Ensure model weights are current
                with profiler.time_section("validation"):
                    # Use EMA weights for validation if available
                    if ema is not None:
                        ema.apply_shadow(model)
                    try:
                        ZImageTrainer._generate_validation_image(
                            model=model,
                            training_spec=training_spec,
                            training_state=training_state,
                        )
                    finally:
                        if ema is not None:
                            ema.restore(model)

            # Save checkpoint periodically
            if training_state.should_save(training_spec):
                deferred_sync.force_sync()  # Ensure all updates complete before checkpoint
                with profiler.time_section("checkpoint"):
                    training_state.save(model, training_spec)

        # Apply any remaining accumulated gradients before saving
        if accumulation_steps > 1 and accumulated_grads is not None and accumulation_count > 0:
            logger.warning(
                f"Applying partial gradient accumulation ({accumulation_count}/{accumulation_steps} steps). "
                f"Consider adjusting batch count to be divisible by gradient_accumulation_steps."
            )
            # Average the partial accumulated gradients (scale creates new dict, so free old one)
            old_accumulated_grads = accumulated_grads
            accumulated_grads = ZImageTrainer._scale_grads(accumulated_grads, 1.0 / accumulation_count)
            del old_accumulated_grads  # Free unscaled gradients
            training_state.optimizer.update(model=model, gradients=accumulated_grads)
            del accumulated_grads

        # Force final sync before saving
        deferred_sync.force_sync()

        # Log sync statistics
        sync_stats = deferred_sync.get_stats()
        logger.info(
            f"Sync efficiency: {sync_stats['efficiency']:.1f}% deferred ({sync_stats['total_deferred']} skipped / {sync_stats['total_syncs']} performed)"
        )

        # Log NaN gradient statistics if any occurred
        if total_nan_skips > 0:
            total_steps = training_state.iterator.num_iterations
            nan_rate = (total_nan_skips / total_steps) * 100 if total_steps > 0 else 0
            logger.warning(
                f"{total_nan_skips} NaN/Inf gradient updates skipped ({nan_rate:.1f}% of steps). "
                f"Training completed but model quality may be affected."
            )

        # Log profiler report if enabled
        if enable_profiling:
            logger.info(f"Profiler report:\n{profiler.report()}")

        # Save final state
        training_state.save(model, training_spec)
        logger.info(f"Training complete! Final checkpoint saved to {training_spec.saver.output_path}")

    @staticmethod
    def _generate_validation_image(
        model: ZImageBase,
        training_spec: TrainingSpec,
        training_state: TrainingState,
    ) -> None:
        """Generate a validation image during training."""
        if training_spec.instrumentation is None:
            return

        image = None
        try:
            image = model.generate_image(
                seed=training_spec.seed,
                prompt=training_spec.instrumentation.validation_prompt,
                negative_prompt=training_spec.instrumentation.negative_prompt,
                guidance_scale=training_spec.instrumentation.guidance_scale,
                num_inference_steps=training_spec.steps,
                width=training_spec.width,
                height=training_spec.height,
            )
            image.save(training_state.get_current_validation_image_path(training_spec))
        except (RuntimeError, ValueError, MemoryError) as e:
            # Catch specific exceptions that may occur during validation generation
            # RuntimeError: MLX operation failures
            # ValueError: Invalid parameters
            # MemoryError: OOM during generation
            logger.warning(f"Failed to generate validation image: {type(e).__name__}: {e}")
        finally:
            if image is not None:
                del image

    @staticmethod
    def _accumulate_grads(acc_grads: dict, new_grads: dict) -> dict:
        """Accumulate gradients for gradient accumulation."""
        result = {}
        # Handle all keys from both dictionaries
        all_keys = set(acc_grads.keys()) | set(new_grads.keys())
        for key in all_keys:
            if key in acc_grads and key in new_grads:
                result[key] = acc_grads[key] + new_grads[key]
            elif key in acc_grads:
                result[key] = acc_grads[key]
            else:
                result[key] = new_grads[key]
        return result

    @staticmethod
    def _scale_grads(grads: dict, scale: float) -> dict:
        """Scale gradients by a factor."""
        return {k: v * scale for k, v in grads.items()}

    @staticmethod
    def _clip_grad_norm(grads: dict, max_norm: float) -> dict | None:
        """Clip gradients by global norm to prevent gradient explosion.

        Args:
            grads: Dictionary of gradients
            max_norm: Maximum allowed gradient norm

        Returns:
            Clipped gradients (same dict if no clipping needed), or None if
            gradients contain NaN/Inf values (caller should skip optimizer update)
        """
        if max_norm <= 0:
            return grads

        # Compute global gradient norm
        total_norm_sq = mx.array(0.0)
        for g in grads.values():
            total_norm_sq = total_norm_sq + mx.sum(g**2)
        total_norm = mx.sqrt(total_norm_sq)

        # Check for NaN/Inf and skip update if detected
        if not mx.isfinite(total_norm).item():
            logger.warning(f"Non-finite gradient norm detected ({total_norm.item()}). Skipping update.")
            return None

        # Compute clipping coefficient
        clip_coef = max_norm / (total_norm + 1e-6)
        clip_coef = mx.minimum(clip_coef, mx.array(1.0))

        # Apply clipping (no-op if norm is already small enough)
        return {k: v * clip_coef for k, v in grads.items()}
