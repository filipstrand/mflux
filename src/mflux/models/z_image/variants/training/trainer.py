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
from typing import TYPE_CHECKING

import mlx.core as mx
from tqdm import tqdm

from mflux.models.common.config.config import Config

if TYPE_CHECKING:
    from PIL import Image
from mflux.models.z_image.variants.training.modes.full_trainer import FullTrainer
from mflux.models.z_image.variants.training.modes.lora_trainer import LoRATrainer
from mflux.models.z_image.variants.training.optimization.compiled_train_step import CompiledTrainStep
from mflux.models.z_image.variants.training.optimization.deferred_sync import create_synchronizer
from mflux.models.z_image.variants.training.optimization.memory_guard import create_memory_guard
from mflux.models.z_image.variants.training.optimization.memory_monitor import create_memory_monitor
from mflux.models.z_image.variants.training.optimization.z_image_loss import ZImageLoss
from mflux.models.z_image.variants.training.state.training_spec import TrainingMode, TrainingSpec
from mflux.models.z_image.variants.training.state.training_state import TrainingState
from mflux.models.z_image.variants.training.statistics.plotter import Plotter
from mflux.models.z_image.variants.training.statistics.profiler import create_profiler
from mflux.models.z_image.variants.training.validation.clip_scorer import create_clip_scorer
from mflux.models.z_image.variants.training.z_image_base import ZImageBase

logger = logging.getLogger(__name__)

# Training loop constants
#
# MEMORY_CHECK_INTERVAL: Number of training steps between memory monitoring checks.
# Set to 100 as a balance between monitoring granularity and overhead:
# - Too frequent (e.g., 10): Adds unnecessary overhead from memory queries
# - Too infrequent (e.g., 1000): May miss memory spikes that cause OOM
# - 100 steps: ~1-2 seconds between checks at typical training speeds
MEMORY_CHECK_INTERVAL = 100


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
        # Disable compilation by default - Batch objects contain non-array members
        # (rng, examples list) that mx.compile() cannot trace through.
        # TODO: Refactor to pass only arrays to the compiled function
        enable_compilation = getattr(training_spec, "enable_compilation", False)
        compiled_train_step = CompiledTrainStep(model, config, enabled=enable_compilation)
        train_step_function = compiled_train_step

        # Setup deferred synchronization (30-40% sync overhead reduction)
        # Sync every 4 updates by default, or per gradient accumulation batch
        sync_interval = getattr(training_spec, "sync_interval", 4)
        deferred_sync = create_synchronizer(sync_interval=sync_interval, adaptive=True)

        # Setup profiler for timing instrumentation
        enable_profiling = getattr(training_spec, "enable_profiling", False)
        profiler = create_profiler(enabled=enable_profiling)

        # Setup memory monitor (detects OOM early, suggests batch size reduction)
        enable_memory_monitoring = training_spec.instrumentation is not None and getattr(
            training_spec.instrumentation, "enable_memory_monitoring", False
        )
        memory_monitor = create_memory_monitor(enabled=enable_memory_monitoring)

        # Setup memory guard (hard limit with auto-pause at 340GB)
        # This is always enabled as a safety net - it only triggers when memory
        # actually exceeds the limit, so there's no overhead during normal operation
        memory_guard = create_memory_guard(
            enabled=True,
            hard_limit_gb=340.0,
            resume_threshold_gb=300.0,
        )

        # Setup CLIP scorer for validation (measures prompt-image alignment)
        enable_clip_score = training_spec.instrumentation is not None and getattr(
            training_spec.instrumentation, "compute_clip_score", False
        )
        clip_scorer = create_clip_scorer(enabled=enable_clip_score)

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

        # Cache validation batch to avoid re-fetching on each validation step
        # Validation data doesn't change during training, so caching is safe
        cached_validation_batch = None

        # Main training loop
        for batch in batches:
            # Check memory and auto-pause if over 340GB limit
            if memory_guard is not None:
                memory_guard.check_and_wait()

            # Compute loss and gradients
            with profiler.time_section("forward"):
                loss, grads = train_step_function(batch)

            # Evaluate only loss for logging - keep grads lazy for accumulation optimization
            # MLX can fuse gradient operations across accumulation steps when grads stay lazy.
            # The accumulated_grads are evaluated once accumulation is complete (see below).
            mx.eval(loss)

            # Periodic memory monitoring
            if memory_monitor is not None and training_state.iterator.num_iterations % MEMORY_CHECK_INTERVAL == 0:
                snapshot = memory_monitor.check()
                if snapshot.status == "critical":
                    logger.warning(
                        f"Memory critical: {snapshot.utilization:.1%} ({snapshot.active_gb:.1f}GB / {snapshot.available_gb:.1f}GB). "
                        f"Consider reducing batch size or enabling gradient checkpointing."
                    )
                elif snapshot.status == "warning":
                    logger.info(
                        f"Memory warning: {snapshot.utilization:.1%} ({snapshot.active_gb:.1f}GB / {snapshot.available_gb:.1f}GB)"
                    )

            # Gradient accumulation
            if accumulation_steps > 1:
                if accumulated_grads is None:
                    # First step: move grads to accumulated_grads (same reference)
                    accumulated_grads = grads
                else:
                    # Subsequent steps: in-place accumulation to avoid doubling memory
                    # Previous approach created a new dict which briefly doubled memory:
                    # accumulated_grads = {k: acc[k] + grads[k] for k in acc}  # Creates new dict!
                    # Fix: accumulate in-place by updating values directly
                    for k in accumulated_grads:
                        accumulated_grads[k] = accumulated_grads[k] + grads[k]
                    del grads  # Free immediately after accumulation
                accumulation_count += 1

                if accumulation_count >= accumulation_steps:
                    # Force evaluation only when accumulation is complete
                    # This prevents lazy graph buildup while minimizing sync overhead
                    mx.eval(accumulated_grads)
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

                # Use cached validation batch to avoid re-fetching on each validation step
                if cached_validation_batch is None:
                    cached_validation_batch = training_state.iterator.get_validation_batch()

                try:
                    with profiler.time_section("validation"):
                        validation_loss = ZImageLoss.compute_validation_loss(model, config, cached_validation_batch)
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
                    # Note: validation_batch is cached, don't delete it
                    if "validation_loss" in locals():
                        del validation_loss

            # Generate validation image periodically
            if training_state.should_generate_image(training_spec):
                deferred_sync.force_sync()  # Ensure model weights are current
                # Log memory status before validation
                if memory_monitor is not None:
                    memory_monitor.log_status()
                with profiler.time_section("validation"):
                    # Use EMA weights for validation if available
                    if ema is not None:
                        ema.apply_shadow(model)
                    try:
                        validation_image = ZImageTrainer._generate_validation_image(
                            model=model,
                            training_spec=training_spec,
                            training_state=training_state,
                        )
                        # Compute CLIP score if enabled and image was generated
                        if validation_image is not None and clip_scorer is not None:
                            try:
                                clip_score = clip_scorer.compute_score(
                                    validation_image,
                                    training_spec.instrumentation.validation_prompt,
                                )
                                training_state.statistics.append_values(
                                    step=training_state.iterator.num_iterations,
                                    clip_score=clip_score,
                                )
                                logger.info(f"CLIP score: {clip_score:.2f}")
                            except (RuntimeError, ValueError, OSError) as e:
                                logger.warning(f"Failed to compute CLIP score: {e}")
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

        # Log memory monitor statistics if enabled
        if memory_monitor is not None:
            mem_stats = memory_monitor.get_stats()
            logger.info(
                f"Memory stats: {mem_stats['check_count']} checks, "
                f"{mem_stats['warning_count']} warnings, {mem_stats['critical_count']} critical, "
                f"peak utilization: {mem_stats['peak_utilization']:.1%}"
            )

        # Log memory guard statistics
        if memory_guard is not None:
            guard_stats = memory_guard.get_stats()
            if guard_stats["pause_count"] > 0:
                logger.warning(
                    f"Memory guard paused training {guard_stats['pause_count']} times, "
                    f"total wait: {guard_stats['total_wait_time_seconds']:.1f}s"
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
    ) -> "Image.Image | None":
        """Generate a validation image during training.

        Returns:
            PIL Image if successful, None otherwise. The image is also saved to disk.
        """
        if training_spec.instrumentation is None:
            return None

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
            return image  # Return image for CLIP scoring
        except (RuntimeError, ValueError, MemoryError) as e:
            # Catch specific exceptions that may occur during validation generation
            # RuntimeError: MLX operation failures
            # ValueError: Invalid parameters
            # MemoryError: OOM during generation
            logger.warning(f"Failed to generate validation image: {type(e).__name__}: {e}")
            return None

    @staticmethod
    def _accumulate_grads(acc_grads: dict[str, mx.array], new_grads: dict[str, mx.array]) -> dict[str, mx.array]:
        """Accumulate gradients for gradient accumulation.

        Assumes gradient structures are static across training steps (same keys).
        """
        # Gradient structures are static - no need for set union
        return {k: acc_grads[k] + new_grads[k] for k in acc_grads}

    @staticmethod
    def _scale_grads(grads: dict[str, mx.array], scale: float) -> dict[str, mx.array]:
        """Scale gradients by a factor."""
        return {k: v * scale for k, v in grads.items()}

    @staticmethod
    def _clip_grad_norm(grads: dict[str, mx.array], max_norm: float) -> dict[str, mx.array] | None:
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
