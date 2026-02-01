"""Z-Image training initializer.

Handles initialization of all training resources:
- Model loading (with optional LoRA layers)
- Dataset preparation (with pre-computed embeddings)
- Optimizer setup
- Training state management
"""

import logging
from pathlib import Path
from typing import Optional

import mlx.core.random as random

from mflux.models.common.config.config import Config
from mflux.models.common.config.model_config import ModelConfig
from mflux.models.z_image.variants.training.dataset.dataset import Dataset
from mflux.models.z_image.variants.training.dataset.iterator import Iterator
from mflux.models.z_image.variants.training.lora_layers.lora_layers import ZImageLoRALayers
from mflux.models.z_image.variants.training.optimization.ema import create_ema
from mflux.models.z_image.variants.training.optimization.lr_scheduler import create_scheduler
from mflux.models.z_image.variants.training.optimization.optimizer import Optimizer
from mflux.models.z_image.variants.training.state.training_spec import TrainingMode, TrainingSpec
from mflux.models.z_image.variants.training.state.training_state import TrainingState
from mflux.models.z_image.variants.training.statistics.statistics import Statistics
from mflux.models.z_image.variants.training.z_image_base import ZImageBase

logger = logging.getLogger(__name__)

# Memory estimation constants (in GB) for Z-Image 6B model
# Based on Z-Image-Base transformer with 6B parameters
# Updated 2026-01 for 512GB Mac Studio target configuration
#
# Derivation:
# - MODEL_BASE_MEMORY: 6B params × 2 bytes (bf16) = 12GB
# - OPTIMIZER_FULL: Gradients (12GB) + AdamW state 2×12GB (m,v) = 36GB + margin = 48GB
# - PER_EXAMPLE: ~2MB average for 512×512 latents + text embeddings
# - ACTIVATIONS_FULL: ~15GB per batch item for full fine-tuning (all layers need gradients)
# - ACTIVATIONS_LORA: ~3GB per batch item for LoRA (only adapter params need gradients)
# - LORA_PARAMS: Typical LoRA adds ~100M trainable params (rank 64, all attention)
# - MLX_OVERHEAD: 1.5× for graph compilation buffers and peak memory spikes
#
# For LoRA training with quantized base model (INT8), activation memory is further reduced.
# See memory_optimizer.py for detailed per-mode estimates.
MODEL_BASE_MEMORY_GB = 12.0  # 6B params at bf16
OPTIMIZER_FULL_MEMORY_GB = 48.0  # Gradients + optimizer state for full fine-tuning
PER_EXAMPLE_MEMORY_MB = 2.0  # Approximate memory per encoded example in MB
PER_BATCH_ACTIVATIONS_FULL_GB = 15.0  # Memory for activations per batch item (full fine-tuning)
PER_BATCH_ACTIVATIONS_LORA_GB = 3.0  # Memory for activations per batch item (LoRA mode)
LORA_PARAMS_ESTIMATE = 100_000_000  # ~100M LoRA params typical configuration
MLX_OVERHEAD_FACTOR = 1.5  # MLX graph overhead and peak memory spikes during backprop


class ZImageTrainingInitializer:
    """Initializes all resources needed for Z-Image training."""

    # Cached system memory value (memoized on first call)
    _cached_system_memory_gb: Optional[float] = None

    @staticmethod
    def initialize(
        config_path: str | None,
        checkpoint_path: str | None,
    ) -> tuple[ZImageBase, Config, TrainingSpec, TrainingState]:
        """Initialize training from config or checkpoint.

        Args:
            config_path: Path to training config JSON file
            checkpoint_path: Path to checkpoint ZIP file (for resuming)

        Returns:
            Tuple of (model, config, training_spec, training_state)
        """
        # Resolve training specification
        training_spec = TrainingSpec.resolve(
            config_path=config_path,
            checkpoint_path=checkpoint_path,
        )

        # Set random seed for reproducibility
        random.seed(training_spec.seed)

        # Load the model
        logger.info("Loading Z-Image model...")
        model_config = ModelConfig.z_image_base()
        model = ZImageBase(
            model_config=model_config,
            quantize=training_spec.quantize,
        )

        # Create inference config for validation
        config = Config(
            model_config=model_config,
            num_inference_steps=training_spec.steps,
            width=training_spec.width,
            height=training_spec.height,
            guidance=training_spec.guidance,
        )

        # Apply LoRA layers if in LoRA mode
        if training_spec.mode == TrainingMode.LORA:
            logger.info("Applying LoRA layers to transformer...")
            ZImageLoRALayers.from_spec(model=model, training_spec=training_spec)

        # Create optimizer
        logger.info("Setting up optimizer...")
        optimizer = Optimizer.from_spec(training_spec)

        # Prepare dataset with pre-computed embeddings
        logger.info("Preparing dataset (this may take a while for large datasets)...")
        dataset_config = training_spec.dataset

        # Resolve and validate cache path if provided
        cache_path: Optional[Path] = None
        if dataset_config and dataset_config.cache_path is not None:
            cache_path = Path(dataset_config.cache_path).expanduser().resolve()
            # Ensure parent directory exists or can be created
            if not cache_path.parent.exists():
                logger.info("Creating cache directory: %s", cache_path.parent)
                cache_path.parent.mkdir(parents=True, exist_ok=True)

        dataset = Dataset.prepare_dataset(
            model=model,
            raw_data=training_spec.examples,
            width=training_spec.width,
            height=training_spec.height,
            enable_augmentation=dataset_config.enable_augmentation if dataset_config else True,
            repeat_count=dataset_config.repeat_count if dataset_config else 1,
            random_crop=dataset_config.random_crop if dataset_config else False,
            seed=training_spec.seed,
            cache_path=cache_path,
        )
        logger.info("Dataset prepared: %d examples", dataset.size())

        # Create iterator
        iterator = Iterator.from_spec(
            training_spec=training_spec,
            dataset=dataset,
        )

        # Setup statistics tracking
        statistics = Statistics.from_spec(training_spec=training_spec)

        # Create EMA if enabled
        ema = None
        if training_spec.ema is not None and training_spec.ema.enabled:
            logger.info("Setting up EMA for weight smoothing...")
            ema = create_ema(
                model=model,
                enabled=True,
                decay=training_spec.ema.decay,
            )

        # Create LR scheduler if specified
        scheduler = None
        scheduler_type = training_spec.optimizer.scheduler_type
        if scheduler_type and scheduler_type.lower() != "constant":
            total_steps = iterator.total_number_of_steps()
            warmup_steps = training_spec.optimizer.warmup_steps
            if training_spec.optimizer.warmup_ratio > 0:
                warmup_steps = int(total_steps * training_spec.optimizer.warmup_ratio)
            logger.info("Setting up %s LR scheduler...", scheduler_type)
            scheduler = create_scheduler(
                name=scheduler_type,
                optimizer=optimizer.optimizer,
                initial_lr=training_spec.optimizer.learning_rate,
                total_steps=total_steps,
                warmup_steps=warmup_steps,
                min_lr=training_spec.optimizer.min_lr,
                pct_start=training_spec.optimizer.pct_start,
            )

        # Create training state
        training_state = TrainingState(
            optimizer=optimizer,
            iterator=iterator,
            statistics=statistics,
            ema=ema,
            scheduler=scheduler,
        )

        logger.info("=== Training Configuration ===")
        logger.info("Mode: %s", training_spec.mode.value)
        logger.info("Model: Z-Image-Base")
        logger.info("Quantization: %s", training_spec.quantize or "None (full precision)")
        logger.info("Resolution: %dx%d", training_spec.width, training_spec.height)
        logger.info("Steps: %d", training_spec.steps)
        logger.info("Guidance: %s", training_spec.guidance)
        logger.info("Batch size: %d", training_spec.training_loop.batch_size)
        logger.info("Epochs: %d", training_spec.training_loop.num_epochs)
        logger.info("Learning rate: %s", training_spec.optimizer.learning_rate)
        logger.info("LR Scheduler: %s", training_spec.optimizer.scheduler_type)
        ema_status = (
            f"enabled (decay={training_spec.ema.decay})"
            if training_spec.ema and training_spec.ema.enabled
            else "disabled"
        )
        logger.info("EMA: %s", ema_status)
        logger.info("Total training examples: %d", dataset.size())
        logger.info("Output path: %s", training_spec.saver.output_path)
        logger.info("=" * 30)

        return model, config, training_spec, training_state

    @staticmethod
    def estimate_memory_usage(training_spec: TrainingSpec) -> dict[str, float]:
        """Estimate memory usage for the training configuration.

        Uses module-level constants for consistent memory estimates.

        Returns:
            Dictionary with memory estimates in GB
        """
        estimates = {}

        # Base model
        estimates["model"] = MODEL_BASE_MEMORY_GB

        # Dataset (uses per-example estimate)
        num_examples = len(training_spec.examples)
        estimates["dataset"] = (num_examples * PER_EXAMPLE_MEMORY_MB) / 1024

        # Optimizer state (for AdamW: 2x model size for momentum)
        if training_spec.mode == TrainingMode.LORA:
            # LoRA parameters are much smaller
            lora_params = LORA_PARAMS_ESTIMATE if training_spec.lora_layers else 0
            estimates["optimizer"] = (lora_params * 2 * 2) / (1024**3)  # fp16 + 2x for Adam
        else:
            estimates["optimizer"] = OPTIMIZER_FULL_MEMORY_GB

        # Activations (batch dependent, varies by training mode)
        batch_size = training_spec.training_loop.batch_size
        activation_per_batch = (
            PER_BATCH_ACTIVATIONS_LORA_GB if training_spec.mode == TrainingMode.LORA else PER_BATCH_ACTIVATIONS_FULL_GB
        )
        estimates["activations"] = batch_size * activation_per_batch

        # Total estimate with MLX overhead factor for peak memory spikes
        subtotal = sum(estimates.values())
        estimates["mlx_overhead"] = subtotal * (MLX_OVERHEAD_FACTOR - 1.0)
        estimates["total"] = subtotal * MLX_OVERHEAD_FACTOR

        return estimates

    @classmethod
    def get_system_memory_gb(cls) -> float | None:
        """Get total system memory in GB (memoized).

        Supports macOS (sysctl) and Linux (/proc/meminfo).
        Returns None on unsupported platforms or detection failure.

        Result is cached after first call to avoid repeated subprocess calls.

        Returns:
            Total system memory in GB, or None if detection fails.
        """
        # Return cached value if available
        if cls._cached_system_memory_gb is not None:
            return cls._cached_system_memory_gb

        import platform
        import subprocess

        system = platform.system()
        result_gb: float | None = None

        try:
            if system == "Darwin":  # macOS
                result = subprocess.run(
                    ["sysctl", "-n", "hw.memsize"],
                    capture_output=True,
                    text=True,
                    timeout=5,
                    shell=False,  # Explicit: prevent shell injection
                )
                if result.returncode == 0:
                    result_gb = int(result.stdout.strip()) / (1024**3)
            elif system == "Linux":
                with open("/proc/meminfo", "r") as f:
                    for line in f:
                        if line.startswith("MemTotal:"):
                            # MemTotal is in kB
                            kb = int(line.split()[1])
                            result_gb = kb / (1024**2)
                            break
            # Windows and other platforms not supported
        except (subprocess.TimeoutExpired, ValueError, OSError, IOError):
            pass

        # Cache the result (even if None, to avoid repeated failed lookups)
        cls._cached_system_memory_gb = result_gb
        return result_gb

    @staticmethod
    def suggest_batch_size(
        training_spec: TrainingSpec,
        safety_margin: float = 0.8,
    ) -> int:
        """Suggest optimal batch size based on available system memory.

        Uses memory estimation to find the largest batch size that fits
        within the available memory with a safety margin.

        Args:
            training_spec: Training specification (batch_size will be varied)
            safety_margin: Fraction of memory to use (default 0.8 = 80%)

        Returns:
            Suggested batch size (1-32), or 1 if detection fails.

        Raises:
            ValueError: If safety_margin is not in valid range (0, 1]
        """
        # Validate safety_margin
        if not (0 < safety_margin <= 1.0):
            raise ValueError(f"safety_margin must be in range (0, 1], got {safety_margin}")

        # Validate training spec has examples
        if not training_spec.examples:
            logger.warning("No examples in training spec, defaulting to batch_size=1")
            return 1

        total_memory_gb = ZImageTrainingInitializer.get_system_memory_gb()

        if total_memory_gb is None:
            logger.warning("Could not detect system memory, defaulting to batch_size=1")
            return 1

        available_memory_gb = total_memory_gb * safety_margin
        logger.info("Detected %.1fGB system memory, using %.1fGB for training", total_memory_gb, available_memory_gb)

        # Pre-compute static memory components (same for all batch sizes)
        dataset_memory = (len(training_spec.examples) * PER_EXAMPLE_MEMORY_MB) / 1024
        if training_spec.mode == TrainingMode.LORA:
            lora_params = LORA_PARAMS_ESTIMATE if training_spec.lora_layers else 0
            optimizer_memory = (lora_params * 2 * 2) / (1024**3)
            activation_per_batch = PER_BATCH_ACTIVATIONS_LORA_GB
        else:
            optimizer_memory = OPTIMIZER_FULL_MEMORY_GB
            activation_per_batch = PER_BATCH_ACTIVATIONS_FULL_GB

        static_memory = MODEL_BASE_MEMORY_GB + dataset_memory + optimizer_memory

        # Test batch sizes from largest to smallest
        candidate_batch_sizes = [32, 16, 8, 4, 2, 1]

        for batch_size in candidate_batch_sizes:
            total_estimate = static_memory + (batch_size * activation_per_batch)

            if total_estimate <= available_memory_gb:
                logger.info("Suggested batch_size=%d (estimated %.1fGB)", batch_size, total_estimate)
                return batch_size

        logger.warning("Even batch_size=1 exceeds memory estimates, proceeding anyway")
        return 1

    @staticmethod
    def auto_tune(
        training_spec: TrainingSpec,
        model: ZImageBase | None = None,
        config: Config | None = None,
        run_test_forward: bool = False,
        safety_margin: float = 0.8,
    ) -> dict:
        """Auto-tune training parameters based on system and model characteristics.

        This method analyzes the system memory, model configuration, and optionally
        runs test forward passes to determine optimal training parameters.

        Args:
            training_spec: Training specification to tune
            model: Optional model for test forward passes
            config: Optional config for test forward passes
            run_test_forward: Whether to run actual forward passes to find max batch size
            safety_margin: Memory safety margin (0.8 = 80%)

        Returns:
            Dictionary with suggested parameters:
            - batch_size: Optimal batch size for system
            - learning_rate: Scaled learning rate based on batch size
            - gradient_accumulation_steps: Suggested accumulation steps
            - sync_interval: Suggested sync interval for deferred sync
            - memory_estimate_gb: Total estimated memory usage

        Usage:
            suggestions = ZImageTrainingInitializer.auto_tune(training_spec)
            print(f"Suggested batch_size: {suggestions['batch_size']}")
            print(f"Suggested learning_rate: {suggestions['learning_rate']}")
        """

        suggestions = {}

        # Get system memory
        total_memory_gb = ZImageTrainingInitializer.get_system_memory_gb()
        if total_memory_gb is None:
            total_memory_gb = 64.0  # Default assumption
            logger.warning("Could not detect system memory, assuming 64GB")

        # Step 1: Find optimal batch size
        if run_test_forward and model is not None and config is not None:
            # Run actual forward passes to find max batch size
            suggestions["batch_size"] = ZImageTrainingInitializer._probe_max_batch_size(
                model=model,
                config=config,
                training_spec=training_spec,
            )
        else:
            # Use memory estimation
            suggestions["batch_size"] = ZImageTrainingInitializer.suggest_batch_size(
                training_spec=training_spec,
                safety_margin=safety_margin,
            )

        # Step 2: Scale learning rate based on batch size
        # Use square root scaling: lr_new = lr_base * sqrt(batch_size / base_batch_size)
        # Reference batch size of 8 is typical for transformer training with lr ~1e-4
        # Square root scaling provides stability for larger batches while maintaining convergence
        base_lr = training_spec.optimizer.learning_rate
        base_batch_size = 8
        actual_batch_size = suggestions["batch_size"]

        # Square root scaling provides good balance between stability and convergence
        lr_scale = (actual_batch_size / base_batch_size) ** 0.5
        suggestions["learning_rate"] = base_lr * lr_scale

        # Step 3: Suggest gradient accumulation for effective batch size
        # Target effective batch size of 8-16 for stable training
        target_effective_batch = 8
        if actual_batch_size < target_effective_batch:
            suggestions["gradient_accumulation_steps"] = max(1, target_effective_batch // actual_batch_size)
        else:
            suggestions["gradient_accumulation_steps"] = 1

        # Step 4: Suggest sync interval based on batch size
        # Larger batches can afford less frequent syncs
        if actual_batch_size >= 8:
            suggestions["sync_interval"] = 8
        elif actual_batch_size >= 4:
            suggestions["sync_interval"] = 4
        else:
            suggestions["sync_interval"] = 2

        # Step 5: Memory estimate
        estimates = ZImageTrainingInitializer.estimate_memory_usage(training_spec)
        # Adjust for actual batch size based on training mode
        activation_per_batch = (
            PER_BATCH_ACTIVATIONS_LORA_GB if training_spec.mode == TrainingMode.LORA else PER_BATCH_ACTIVATIONS_FULL_GB
        )
        estimates["activations"] = actual_batch_size * activation_per_batch
        estimates["total"] = (
            estimates["model"] + estimates["dataset"] + estimates["optimizer"] + estimates["activations"]
        ) * MLX_OVERHEAD_FACTOR
        suggestions["memory_estimate_gb"] = estimates["total"]

        # Step 6: Calculate effective batch size
        suggestions["effective_batch_size"] = actual_batch_size * suggestions["gradient_accumulation_steps"]

        # Log summary
        logger.info("=== Auto-Tune Results ===")
        logger.info("System memory: %.1fGB", total_memory_gb)
        logger.info("Suggested batch_size: %d", suggestions["batch_size"])
        logger.info("Suggested learning_rate: %.2e", suggestions["learning_rate"])
        logger.info("Suggested gradient_accumulation_steps: %d", suggestions["gradient_accumulation_steps"])
        logger.info("Effective batch size: %d", suggestions["effective_batch_size"])
        logger.info("Suggested sync_interval: %d", suggestions["sync_interval"])
        logger.info("Estimated memory usage: %.1fGB", suggestions["memory_estimate_gb"])
        logger.info("=" * 25)

        return suggestions

    @staticmethod
    def _probe_max_batch_size(
        model: ZImageBase,
        config: Config,
        training_spec: TrainingSpec,
    ) -> int:
        """Probe for maximum batch size by running test forward passes.

        Tries progressively smaller batch sizes until one succeeds.

        Args:
            model: Model to test
            config: Config with scheduler
            training_spec: Training spec with examples

        Returns:
            Maximum working batch size
        """
        import random

        import mlx.core as mx

        from mflux.models.z_image.variants.training.dataset.batch import Batch
        from mflux.models.z_image.variants.training.optimization.z_image_loss import ZImageLoss

        # Create a simple test batch from first example
        if not training_spec.examples:
            return 1

        # Prepare a minimal test dataset
        test_dataset = Dataset.prepare_dataset(
            model=model,
            raw_data=training_spec.examples[: min(16, len(training_spec.examples))],
            width=training_spec.width,
            height=training_spec.height,
            enable_augmentation=False,
            repeat_count=1,
            random_crop=False,
        )

        if test_dataset.size() == 0:
            return 1

        # Test batch sizes from largest to smallest
        candidate_sizes = [16, 12, 8, 6, 4, 2, 1]

        for batch_size in candidate_sizes:
            if batch_size > test_dataset.size():
                continue

            try:
                # Create test batch
                test_examples = test_dataset.examples[:batch_size]
                test_batch = Batch(
                    examples=test_examples,
                    rng=random.Random(42),
                )

                # Run forward pass
                loss = ZImageLoss.compute_loss(model, config, test_batch)
                mx.synchronize()  # Force computation

                # If we got here, this batch size works
                logger.info("Probed batch_size=%d: Success (loss=%.4f)", batch_size, float(loss))

                # Clean up
                del loss, test_batch
                mx.synchronize()

                return batch_size

            except Exception as e:  # noqa: BLE001 - Intentional: probing memory limits with various batch sizes
                logger.debug("Probed batch_size=%d: Failed (%s)", batch_size, type(e).__name__)
                mx.synchronize()
                continue

        # If we get here, even batch_size=1 failed - verify it's truly unusable
        logger.warning("All batch sizes failed during probing. Verifying batch_size=1...")
        try:
            # Final verification for batch_size=1
            if test_dataset.size() >= 1:
                test_examples = test_dataset.examples[:1]
                test_batch = Batch(
                    examples=test_examples,
                    rng=random.Random(42),
                )
                loss = ZImageLoss.compute_loss(model, config, test_batch)
                mx.synchronize()
                del loss, test_batch
                logger.info("Verification passed: batch_size=1 works")
                return 1
        except Exception as e:  # noqa: BLE001
            raise RuntimeError(
                f"Cannot train: even batch_size=1 causes failure. "
                f"Reduce image dimensions or model size. Error: {type(e).__name__}: {e}"
            ) from e

        return 1
