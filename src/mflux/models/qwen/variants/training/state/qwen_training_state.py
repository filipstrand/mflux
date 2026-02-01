"""
Qwen-Image Training State Management.

Handles training state including:
- Iterator state (current epoch, batch)
- Optimizer state
- EMA state
- Statistics (loss history)
- Checkpoint saving/loading
"""

import datetime
import json
import tempfile
import zipfile
from pathlib import Path
from random import Random
from typing import (
    TYPE_CHECKING,
    Any,
    Iterator as TypingIterator,
)

import mlx.core as mx

from mflux.models.qwen.variants.training.dataset.qwen_batch import QwenBatch
from mflux.models.qwen.variants.training.dataset.qwen_dataset import QwenDataset
from mflux.models.qwen.variants.training.optimization.ema import EMAModel, create_ema
from mflux.models.qwen.variants.training.optimization.gradient_accumulator import (
    GradientAccumulator,
    create_accumulator,
)
from mflux.models.qwen.variants.training.optimization.lr_scheduler import (
    LRScheduler,
    create_scheduler,
)
from mflux.models.qwen.variants.training.state.qwen_training_spec import QwenTrainingSpec

if TYPE_CHECKING:
    from mflux.models.qwen.variants.txt2img.qwen_image import QwenImage


# File name constants
CHECKPOINT_FILE = "checkpoint.json"
OPTIMIZER_FILE = "optimizer.safetensors"
ADAPTER_FILE = "adapter.safetensors"
ITERATOR_FILE = "iterator.json"
LOSS_FILE = "loss.json"
CONFIG_FILE = "config.json"
EMA_FILE = "ema.safetensors"
LR_SCHEDULER_FILE = "lr_scheduler.json"
GRAD_ACCUMULATOR_FILE = "grad_accumulator.json"


class QwenIterator:
    """
    Training data iterator with state tracking.

    Iterates through epochs and batches while tracking progress
    for checkpointing and resumption.
    """

    def __init__(
        self,
        dataset: QwenDataset,
        batch_size: int,
        seed: int = 42,
        shuffle_first_epoch: bool = True,
    ):
        # Validate dataset is not empty
        if len(dataset.examples) == 0:
            raise ValueError("Dataset is empty! Cannot train on zero examples.")

        # Validate batch_size
        if batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {batch_size}")

        self.dataset = dataset
        self.batch_size = batch_size
        self.seed = seed

        self._rng = Random(seed)
        self._epoch = 0
        self._batch_idx = 0
        self._num_iterations = 0
        self._start_time = datetime.datetime.now()

        # Shuffle dataset for first epoch (otherwise first epoch is unshuffled)
        if shuffle_first_epoch:
            self._shuffle()

    @property
    def epoch(self) -> int:
        return self._epoch

    @property
    def num_iterations(self) -> int:
        return self._num_iterations

    @property
    def start_time(self) -> datetime.datetime:
        return self._start_time

    def __iter__(self) -> TypingIterator[QwenBatch]:
        return self

    def __next__(self) -> QwenBatch:
        """Get next batch."""
        if self._batch_idx >= len(self.dataset) // self.batch_size:
            # End of epoch
            self._epoch += 1
            self._batch_idx = 0
            # Shuffle for next epoch
            self._shuffle()

        # Get batch indices
        start = self._batch_idx * self.batch_size
        end = start + self.batch_size
        examples = self.dataset.examples[start:end]

        self._batch_idx += 1
        self._num_iterations += 1

        return QwenBatch(examples=examples, rng=self._rng)

    def _shuffle(self) -> None:
        """Shuffle dataset examples."""
        self._rng.shuffle(self.dataset.examples)

    def state_dict(self) -> dict[str, Any]:
        """Get iterator state for checkpointing."""
        return {
            "epoch": self._epoch,
            "batch_idx": self._batch_idx,
            "num_iterations": self._num_iterations,
            "seed": self.seed,
            "rng_state": self._serialize_rng_state(self._rng.getstate()),
        }

    @staticmethod
    def _serialize_rng_state(state: tuple) -> dict[str, Any]:
        """
        Serialize Python Random.getstate() to JSON-compatible dict.

        Random.getstate() returns (version, internal_state_tuple, gauss_next)
        where internal_state_tuple contains 625 integers.
        """
        version, internal_state, gauss_next = state
        return {
            "version": version,
            "internal_state": list(internal_state),
            "gauss_next": gauss_next,
        }

    @staticmethod
    def _deserialize_rng_state(state_dict: dict[str, Any]) -> tuple:
        """Deserialize JSON dict back to Random.setstate() format."""
        return (
            state_dict["version"],
            tuple(state_dict["internal_state"]),
            state_dict["gauss_next"],
        )

    def load_state_dict(self, state: dict[str, Any]) -> None:
        """Load iterator state from checkpoint."""
        self._epoch = state["epoch"]
        self._batch_idx = state["batch_idx"]
        self._num_iterations = state["num_iterations"]
        rng_state = state["rng_state"]
        # Handle both old format (list) and new format (dict)
        if isinstance(rng_state, dict):
            rng_state = self._deserialize_rng_state(rng_state)
        elif isinstance(rng_state, list):
            # Legacy format: [version, [ints...], gauss_next]
            # Convert to tuple format expected by Random.setstate()
            rng_state = (rng_state[0], tuple(rng_state[1]), rng_state[2])
        self._rng.setstate(rng_state)

    def save(self, path: Path) -> None:
        """Save iterator state to file."""
        with open(path, "w") as f:
            state = self.state_dict()
            json.dump(state, f, indent=2)


class QwenStatistics:
    """Training statistics tracking."""

    def __init__(self):
        self._losses: list[float] = []
        self._iterations: list[int] = []

    def add_loss(self, iteration: int, loss: float) -> None:
        """Record a loss value."""
        self._iterations.append(iteration)
        self._losses.append(loss)

    @property
    def losses(self) -> list[float]:
        return self._losses

    @property
    def iterations(self) -> list[int]:
        return self._iterations

    @property
    def last_loss(self) -> float | None:
        return self._losses[-1] if self._losses else None

    def state_dict(self) -> dict[str, Any]:
        return {
            "losses": self._losses,
            "iterations": self._iterations,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        """Load statistics state from checkpoint with validation."""
        losses = state.get("losses")
        iterations = state.get("iterations")

        # Validate structure
        if not isinstance(losses, list):
            raise ValueError(f"Invalid statistics format: losses must be list, got {type(losses)}")
        if not isinstance(iterations, list):
            raise ValueError(f"Invalid statistics format: iterations must be list, got {type(iterations)}")
        if len(losses) != len(iterations):
            raise ValueError(f"Mismatched statistics lengths: {len(losses)} losses vs {len(iterations)} iterations")

        self._losses = losses
        self._iterations = iterations

    def save(self, path: Path) -> None:
        with open(path, "w") as f:
            json.dump(self.state_dict(), f, indent=2)


class QwenOptimizer:
    """Wrapper for MLX optimizer with state management."""

    def __init__(self, optimizer: Any):
        self.optimizer = optimizer

    def save(self, path: Path) -> None:
        from mlx.utils import tree_flatten

        state = tree_flatten(self.optimizer.state)
        mx.save_safetensors(str(path), dict(state))


class QwenTrainingState:
    """
    Complete training state for Qwen-Image.

    Manages all stateful components of training including:
    - Data iterator
    - Optimizer
    - LR scheduler
    - Gradient accumulator
    - EMA model
    - Statistics
    """

    def __init__(
        self,
        iterator: QwenIterator,
        optimizer: QwenOptimizer,
        lr_scheduler: LRScheduler,
        grad_accumulator: GradientAccumulator,
        ema: EMAModel | None,
        statistics: QwenStatistics,
    ):
        self.iterator = iterator
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.grad_accumulator = grad_accumulator
        self.ema = ema
        self.statistics = statistics

    @staticmethod
    def create(
        dataset: QwenDataset,
        spec: QwenTrainingSpec,
        model: "QwenImage",
    ) -> "QwenTrainingState":
        """
        Create training state from spec.

        Args:
            dataset: Prepared training dataset
            spec: Training specification
            model: QwenImage model (for EMA initialization)

        Returns:
            Initialized QwenTrainingState
        """
        import mlx.optimizers as optim

        # Create iterator
        iterator = QwenIterator(
            dataset=dataset,
            batch_size=spec.training_loop.batch_size,
            seed=spec.seed,
        )

        # Create optimizer
        opt_class = getattr(optim, spec.optimizer.name)
        mlx_optimizer = opt_class(
            learning_rate=spec.optimizer.learning_rate,
        )
        if hasattr(mlx_optimizer, "weight_decay"):
            mlx_optimizer.weight_decay = spec.optimizer.weight_decay
        optimizer = QwenOptimizer(mlx_optimizer)

        # Create LR scheduler
        lr_scheduler = create_scheduler(
            name=spec.lr_scheduler.name,
            optimizer=mlx_optimizer,
            initial_lr=spec.optimizer.learning_rate,
            total_steps=spec.total_steps,
            warmup_steps=spec.lr_scheduler.warmup_steps,
            min_lr=spec.lr_scheduler.min_lr,
        )

        # Create gradient accumulator
        grad_accumulator = create_accumulator(spec.training_loop.gradient_accumulation_steps)

        # Create EMA
        ema = (
            create_ema(
                model=model,
                enabled=spec.ema.enabled,
                decay=spec.ema.decay,
            )
            if spec.ema.enabled
            else None
        )

        # Create statistics tracker
        statistics = QwenStatistics()

        return QwenTrainingState(
            iterator=iterator,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            grad_accumulator=grad_accumulator,
            ema=ema,
            statistics=statistics,
        )

    def should_save(self, spec: QwenTrainingSpec) -> bool:
        """Check if checkpoint should be saved."""
        return self.iterator.num_iterations % spec.saver.checkpoint_frequency == 0

    def should_plot(self, spec: QwenTrainingSpec) -> bool:
        """Check if loss should be plotted."""
        if spec.instrumentation is None:
            return False
        return self.iterator.num_iterations % spec.instrumentation.plot_frequency == 0

    def should_validate(self, spec: QwenTrainingSpec) -> bool:
        """Check if validation image should be generated."""
        if spec.instrumentation is None:
            return False
        return self.iterator.num_iterations % spec.instrumentation.generate_image_frequency == 0

    def save_checkpoint(
        self,
        model: "QwenImage",
        spec: QwenTrainingSpec,
    ) -> Path:
        """
        Save training checkpoint.

        Saves all training state including:
        - Optimizer state
        - LoRA/DoRA adapter weights
        - Iterator state (epoch, batch_idx, RNG)
        - LR scheduler state (step_count, parameters)
        - Gradient accumulator state
        - Loss statistics
        - EMA weights (if enabled)
        - Training config

        Args:
            model: QwenImage model
            spec: Training specification

        Returns:
            Path to saved checkpoint
        """
        from mflux.models.common.lora.layer.adapter_factory import AdapterFactory

        iteration = self.iterator.num_iterations

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Save components
            opt_path = temp_path / f"{iteration:07d}_{OPTIMIZER_FILE}"
            adapter_path = temp_path / f"{iteration:07d}_{ADAPTER_FILE}"
            iter_path = temp_path / f"{iteration:07d}_{ITERATOR_FILE}"
            loss_path = temp_path / f"{iteration:07d}_{LOSS_FILE}"
            config_path = temp_path / f"{iteration:07d}_{CONFIG_FILE}"
            lr_scheduler_path = temp_path / f"{iteration:07d}_{LR_SCHEDULER_FILE}"
            grad_accum_path = temp_path / f"{iteration:07d}_{GRAD_ACCUMULATOR_FILE}"
            checkpoint_path = temp_path / CHECKPOINT_FILE

            self.optimizer.save(opt_path)

            # Save LoRA/DoRA adapter weights
            adapter_path_name = None
            try:
                AdapterFactory.save_adapters(
                    model=model.transformer,
                    path=str(adapter_path),
                    adapter_type=spec.lora_layers.adapter_type,
                )
                adapter_path_name = adapter_path.name
            except ValueError:
                # No adapters to save (might be full fine-tuning)
                pass

            self.iterator.save(iter_path)
            self.statistics.save(loss_path)

            # Save LR scheduler state
            lr_scheduler_state = self.lr_scheduler.state_dict()
            lr_scheduler_state["scheduler_type"] = type(self.lr_scheduler).__name__
            with open(lr_scheduler_path, "w") as f:
                json.dump(lr_scheduler_state, f, indent=2)

            # Save gradient accumulator state
            grad_accum_state = self.grad_accumulator.state_dict()
            with open(grad_accum_path, "w") as f:
                json.dump(grad_accum_state, f, indent=2)

            # Save config
            with open(config_path, "w") as f:
                f.write(spec.to_json())

            # Save EMA if enabled
            ema_path = None
            if self.ema is not None:
                ema_path = temp_path / f"{iteration:07d}_{EMA_FILE}"
                self.ema.save(ema_path)

            # Create checkpoint manifest
            manifest = {
                "metadata": {
                    "iteration": iteration,
                    "epoch": self.iterator.epoch,
                    "timestamp": datetime.datetime.now().isoformat(),
                },
                "files": {
                    "optimizer": opt_path.name,
                    "adapter": adapter_path_name,
                    "iterator": iter_path.name,
                    "loss": loss_path.name,
                    "config": config_path.name,
                    "lr_scheduler": lr_scheduler_path.name,
                    "grad_accumulator": grad_accum_path.name,
                    "ema": ema_path.name if ema_path else None,
                },
            }
            with open(checkpoint_path, "w") as f:
                json.dump(manifest, f, indent=2)

            # Create zip file
            output_dir = Path(spec.saver.output_path) / "_checkpoints"
            output_dir.mkdir(parents=True, exist_ok=True)
            zip_path = output_dir / f"{iteration:07d}_checkpoint.zip"

            with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
                for path in temp_path.iterdir():
                    if path.exists():
                        zipf.write(path, path.name)

            return zip_path

    @staticmethod
    def load_checkpoint(
        checkpoint_path: str | Path,
        dataset: QwenDataset,
        spec: QwenTrainingSpec,
        model: "QwenImage",
    ) -> "QwenTrainingState":
        """
        Load training state from checkpoint.

        Restores all training state including:
        - Iterator state (epoch, batch_idx, RNG)
        - LR scheduler state
        - Gradient accumulator state
        - Loss statistics
        - EMA weights (if enabled)

        IMPORTANT: Adapter weights (LoRA/DoRA) must be loaded separately
        BEFORE calling this method. Use AdapterFactory.load_adapters() to
        load the adapter weights from the checkpoint:

            # Example usage:
            from mflux.models.common.lora.layer.adapter_factory import AdapterFactory

            # First, load adapter weights into the model
            AdapterFactory.load_adapters(
                model=model.transformer,
                path=checkpoint_adapter_path,  # Extract from checkpoint ZIP
            )

            # Then load training state
            state = QwenTrainingState.load_checkpoint(
                checkpoint_path=checkpoint_path,
                dataset=dataset,
                spec=spec,
                model=model,
            )

        Note: MLX optimizer state (momentum, Adam state) is NOT restored.
        This is intentional - many training setups reset optimizer state
        when resuming. The LR scheduler position IS restored to maintain
        the correct learning rate schedule.

        Args:
            checkpoint_path: Path to checkpoint ZIP file
            dataset: Training dataset (must be same as original training)
            spec: Training specification
            model: QwenImage model with adapters already applied and loaded

        Returns:
            Restored QwenTrainingState
        """
        import mlx.optimizers as optim

        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Extract checkpoint
            with zipfile.ZipFile(checkpoint_path, "r") as zipf:
                zipf.extractall(temp_path)

            # Load manifest
            manifest_path = temp_path / CHECKPOINT_FILE
            with open(manifest_path, "r") as f:
                manifest = json.load(f)

            files = manifest["files"]

            # Restore iterator
            iterator = QwenIterator(
                dataset=dataset,
                batch_size=spec.training_loop.batch_size,
                seed=spec.seed,
                shuffle_first_epoch=False,  # Don't shuffle - we'll restore state
            )
            iter_file = temp_path / files["iterator"]
            with open(iter_file, "r") as f:
                iter_state = json.load(f)
            iterator.load_state_dict(iter_state)

            # Create optimizer (state will be fresh, but LR scheduler restores position)
            opt_class = getattr(optim, spec.optimizer.name)
            mlx_optimizer = opt_class(
                learning_rate=spec.optimizer.learning_rate,
            )
            if hasattr(mlx_optimizer, "weight_decay"):
                mlx_optimizer.weight_decay = spec.optimizer.weight_decay
            optimizer = QwenOptimizer(mlx_optimizer)

            # Restore LR scheduler
            lr_file = temp_path / files["lr_scheduler"]
            with open(lr_file, "r") as f:
                lr_state = json.load(f)

            lr_scheduler = create_scheduler(
                name=spec.lr_scheduler.name,
                optimizer=mlx_optimizer,
                initial_lr=spec.optimizer.learning_rate,
                total_steps=spec.total_steps,
                warmup_steps=spec.lr_scheduler.warmup_steps,
                min_lr=spec.lr_scheduler.min_lr,
            )
            lr_scheduler.load_state_dict(lr_state)

            # Restore gradient accumulator
            grad_file = temp_path / files["grad_accumulator"]
            with open(grad_file, "r") as f:
                grad_state = json.load(f)

            grad_accumulator = create_accumulator(spec.training_loop.gradient_accumulation_steps)
            grad_accumulator.load_state_dict(grad_state)

            # Restore statistics
            statistics = QwenStatistics()
            loss_file = temp_path / files["loss"]
            with open(loss_file, "r") as f:
                loss_state = json.load(f)
            statistics.load_state_dict(loss_state)

            # Restore EMA if present
            ema = None
            if spec.ema.enabled and files.get("ema"):
                ema_file = temp_path / files["ema"]
                if ema_file.exists():
                    # Use classmethod to load EMA with saved shadow weights
                    from mflux.models.qwen.variants.training.optimization.ema import EMAModel

                    ema = EMAModel.load(str(ema_file), model=model, decay=spec.ema.decay)
                else:
                    # No saved EMA, create fresh one
                    ema = create_ema(
                        model=model,
                        enabled=True,
                        decay=spec.ema.decay,
                    )

            return QwenTrainingState(
                iterator=iterator,
                optimizer=optimizer,
                lr_scheduler=lr_scheduler,
                grad_accumulator=grad_accumulator,
                ema=ema,
                statistics=statistics,
            )
