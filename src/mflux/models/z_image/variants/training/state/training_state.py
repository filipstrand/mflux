import datetime
import json
import logging
import tempfile
import zipfile
from pathlib import Path
from typing import TYPE_CHECKING

from mflux.models.z_image.variants.training.dataset.iterator import Iterator
from mflux.models.z_image.variants.training.lora_layers.lora_layers import ZImageLoRALayers
from mflux.models.z_image.variants.training.optimization.optimizer import Optimizer
from mflux.models.z_image.variants.training.state.training_spec import TrainingMode, TrainingSpec
from mflux.models.z_image.variants.training.state.zip_util import ZipUtil
from mflux.models.z_image.variants.training.statistics.statistics import Statistics

if TYPE_CHECKING:
    from mflux.models.z_image.variants.training.z_image_base import ZImageBase

logger = logging.getLogger(__name__)

TRAINING_PATH_CHECKPOINTS = "_checkpoints"
TRAINING_FILE_NAME_CHECKPOINT = "checkpoint"

TRAINING_FILE_NAME_LORA_ADAPTER = "adapter"
TRAINING_FILE_NAME_OPTIMIZER = "optimizer"
TRAINING_FILE_NAME_ITERATOR = "iterator"
TRAINING_FILE_NAME_LOSS_FILE = "loss"
TRAINING_FILE_NAME_CONFIG_FILE = "config"
TRAINING_FILE_NAME_MODEL = "model"
TRAINING_FILE_NAME_EMA = "ema"
TRAINING_FILE_NAME_SCHEDULER = "scheduler"

TRAINING_PATH_VALIDATION_IMAGES = "_validation/images/"
TRAINING_FILE_NAME_VALIDATION_IMAGE = "validation_image"

TRAINING_PATH_VALIDATION_PLOT = "_validation/plots/"
TRAINING_FILE_NAME_VALIDATION_LOSS = "loss"
TRAINING_FILE_NAME_EARLY_STOPPING = "early_stopping"


class EarlyStoppingState:
    """State tracker for early stopping during training.

    Monitors validation loss and triggers early stopping when no improvement
    is seen for a specified number of validations (patience).
    """

    def __init__(self, patience: int, min_delta: float):
        """Initialize early stopping state.

        Args:
            patience: Number of validations without improvement before stopping.
            min_delta: Minimum improvement threshold to consider as improvement.
        """
        self.patience = patience
        self.min_delta = min_delta
        self.best_val_loss = float("inf")
        self.patience_counter = 0
        self.should_stop = False

    def check(self, val_loss: float) -> bool:
        """Check if training should stop based on validation loss.

        Args:
            val_loss: Current validation loss.

        Returns:
            True if training should stop, False otherwise.
        """
        # Once triggered, stay triggered - don't mutate state further
        if self.should_stop:
            return True

        if val_loss < self.best_val_loss - self.min_delta:
            self.best_val_loss = val_loss
            self.patience_counter = 0
        else:
            self.patience_counter += 1
            if self.patience_counter >= self.patience:
                self.should_stop = True
                logger.info(f"Early stopping: no improvement for {self.patience} validations")
        return self.should_stop

    def to_dict(self) -> dict:
        """Serialize state to dictionary for checkpointing."""
        return {
            "best_val_loss": self.best_val_loss,
            "patience_counter": self.patience_counter,
            "should_stop": self.should_stop,
        }

    @classmethod
    def from_dict(cls, data: dict, patience: int, min_delta: float) -> "EarlyStoppingState":
        """Restore state from dictionary.

        Args:
            data: Dictionary from to_dict()
            patience: Patience value from EarlyStoppingSpec
            min_delta: Min delta value from EarlyStoppingSpec

        Returns:
            Restored EarlyStoppingState instance.

        Note:
            Values are validated and clamped to sensible ranges to handle
            corrupted checkpoint data gracefully.
        """
        state = cls(patience=patience, min_delta=min_delta)

        # Restore best_val_loss with validation
        best_val_loss = data.get("best_val_loss", float("inf"))
        if not isinstance(best_val_loss, (int, float)) or best_val_loss != best_val_loss:  # NaN check
            best_val_loss = float("inf")
        state.best_val_loss = float(best_val_loss)

        # Restore patience_counter with validation (clamp to valid range)
        patience_counter = data.get("patience_counter", 0)
        if not isinstance(patience_counter, int) or patience_counter < 0:
            patience_counter = 0
        state.patience_counter = min(patience_counter, patience)  # Clamp to max patience

        # Restore should_stop with validation
        should_stop = data.get("should_stop", False)
        state.should_stop = bool(should_stop)

        return state


class TrainingState:
    def __init__(
        self,
        iterator: Iterator,
        optimizer: Optimizer,
        statistics: Statistics,
        ema=None,
        scheduler=None,
        early_stopping: EarlyStoppingState | None = None,
    ):
        self.iterator = iterator
        self.optimizer = optimizer
        self.statistics = statistics
        self.ema = ema  # EMAModel | NoOpEMA | None
        self.scheduler = scheduler  # LRScheduler | None
        self.early_stopping = early_stopping  # EarlyStoppingState | None

    def save(self, model: "ZImageBase", training_spec: TrainingSpec) -> None:
        """Save checkpoint with all training state."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Prepare file paths
            optimizer_path = (
                Path(temp_dir) / f"{self.iterator.num_iterations:07d}_{TRAINING_FILE_NAME_OPTIMIZER}.safetensors"
            )
            iterator_path = Path(temp_dir) / f"{self.iterator.num_iterations:07d}_{TRAINING_FILE_NAME_ITERATOR}.json"
            loss_path = Path(temp_dir) / f"{self.iterator.num_iterations:07d}_{TRAINING_FILE_NAME_LOSS_FILE}.json"
            config_path = Path(temp_dir) / f"{self.iterator.num_iterations:07d}_{TRAINING_FILE_NAME_CONFIG_FILE}.json"
            checkpoint_path = Path(temp_dir) / f"{TRAINING_FILE_NAME_CHECKPOINT}.json"
            paths = [optimizer_path, iterator_path, loss_path, config_path, checkpoint_path]

            checkpoint_files = {
                "config": f"{self.iterator.num_iterations:07d}_{TRAINING_FILE_NAME_CONFIG_FILE}.json",
                "optimizer": f"{self.iterator.num_iterations:07d}_{TRAINING_FILE_NAME_OPTIMIZER}.safetensors",
                "iterator": f"{self.iterator.num_iterations:07d}_{TRAINING_FILE_NAME_ITERATOR}.json",
                "loss": f"{self.iterator.num_iterations:07d}_{TRAINING_FILE_NAME_LOSS_FILE}.json",
            }

            # Save LoRA adapter or full model weights
            if training_spec.mode == TrainingMode.LORA:
                lora_path = (
                    Path(temp_dir) / f"{self.iterator.num_iterations:07d}_{TRAINING_FILE_NAME_LORA_ADAPTER}.safetensors"
                )
                ZImageLoRALayers.save(model.transformer, lora_path, training_spec)
                paths.append(lora_path)
                checkpoint_files["lora_adapter"] = (
                    f"{self.iterator.num_iterations:07d}_{TRAINING_FILE_NAME_LORA_ADAPTER}.safetensors"
                )
            else:
                # For full fine-tuning, save the entire transformer
                model_path = (
                    Path(temp_dir) / f"{self.iterator.num_iterations:07d}_{TRAINING_FILE_NAME_MODEL}.safetensors"
                )
                self._save_full_model(model, model_path)
                paths.append(model_path)
                checkpoint_files["model"] = f"{self.iterator.num_iterations:07d}_{TRAINING_FILE_NAME_MODEL}.safetensors"

            # Save EMA state if enabled
            if self.ema is not None and training_spec.ema is not None and training_spec.ema.enabled:
                ema_path = Path(temp_dir) / f"{self.iterator.num_iterations:07d}_{TRAINING_FILE_NAME_EMA}.safetensors"
                self.ema.save(ema_path)
                paths.append(ema_path)
                checkpoint_files["ema"] = f"{self.iterator.num_iterations:07d}_{TRAINING_FILE_NAME_EMA}.safetensors"

            # Save scheduler state if present
            if self.scheduler is not None:
                scheduler_path = (
                    Path(temp_dir) / f"{self.iterator.num_iterations:07d}_{TRAINING_FILE_NAME_SCHEDULER}.safetensors"
                )
                self.scheduler.save(scheduler_path)
                paths.append(scheduler_path)
                checkpoint_files["scheduler"] = (
                    f"{self.iterator.num_iterations:07d}_{TRAINING_FILE_NAME_SCHEDULER}.safetensors"
                )

            # Save early stopping state if present
            if self.early_stopping is not None:
                early_stopping_path = (
                    Path(temp_dir) / f"{self.iterator.num_iterations:07d}_{TRAINING_FILE_NAME_EARLY_STOPPING}.json"
                )
                with open(early_stopping_path, "w") as f:
                    json.dump(self.early_stopping.to_dict(), f, indent=4)
                paths.append(early_stopping_path)
                checkpoint_files["early_stopping"] = (
                    f"{self.iterator.num_iterations:07d}_{TRAINING_FILE_NAME_EARLY_STOPPING}.json"
                )

            # Save individual components
            self.optimizer.save(optimizer_path)
            self.iterator.save(iterator_path)
            self.statistics.save(loss_path)
            self._save_train_config(config_path, training_spec)

            # Create checkpoint metadata
            checkpoint_data = self._create_checkpoint_data(
                training_spec, self.iterator.start_date_time, checkpoint_files
            )
            with open(checkpoint_path, "w") as json_file:
                json.dump(checkpoint_data, json_file, indent=4)

            # Create zip file
            output_path = Path(training_spec.saver.output_path) / TRAINING_PATH_CHECKPOINTS
            output_path.mkdir(parents=True, exist_ok=True)
            zip_path = output_path / f"{self.iterator.num_iterations:07d}_checkpoint.zip"

            with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
                for file_path in paths:
                    # Use try-except instead of exists() check to avoid TOCTOU race condition
                    try:
                        zipf.write(file_path, file_path.name)
                    except FileNotFoundError:  # noqa: PERF203 - Intentional: TOCTOU safety
                        logger.warning(f"Could not save {file_path.name} to checkpoint - file not found")

        # Prune old checkpoints if configured
        deleted, failed = self._prune_old_checkpoints(training_spec)
        if deleted:
            logger.info(f"Pruned {len(deleted)} old checkpoint(s)")
        if failed:
            logger.warning(f"Failed to prune {len(failed)} checkpoint(s) - check disk space")

    def _prune_old_checkpoints(self, training_spec: TrainingSpec) -> tuple[list[Path], list[Path]]:
        """Remove old checkpoints, keeping only the most recent N.

        Args:
            training_spec: Training specification with saver.keep_last_n_checkpoints

        Returns:
            Tuple of (deleted paths, failed paths) for monitoring pruning health
        """
        keep_last = training_spec.saver.keep_last_n_checkpoints
        if keep_last <= 0:
            return [], []

        checkpoint_dir = Path(training_spec.saver.output_path) / TRAINING_PATH_CHECKPOINTS
        if not checkpoint_dir.exists():
            return [], []

        checkpoints = sorted(checkpoint_dir.glob("*_checkpoint.zip"))
        if len(checkpoints) <= keep_last:
            return [], []

        to_delete = checkpoints[:-keep_last]
        deleted = []
        failed = []
        for path in to_delete:
            try:
                path.unlink()
                deleted.append(path)
                logger.debug(f"Pruned checkpoint: {path.name}")
            except OSError as e:  # noqa: PERF203 - Intentional: continue pruning other checkpoints on failure
                logger.warning(f"Failed to delete {path}: {e}")
                failed.append(path)

        return deleted, failed

    def _save_full_model(self, model: "ZImageBase", path: Path) -> None:
        """Save full transformer model weights for full fine-tuning mode.

        Uses MLX's tree_flatten to extract all transformer parameters and saves
        them in safetensors format. Only the transformer is saved because:
        - VAE is typically frozen during training
        - Text encoder is typically frozen during training

        The weights are prefixed with 'transformer.' to match the model structure
        and enable proper loading during inference.

        Args:
            model: The Z-Image model containing the transformer to save
            path: Destination path for the safetensors file
        """
        import mlx.core as mx
        from mlx.utils import tree_flatten

        # Save transformer weights
        weights = {}
        for entry in tree_flatten(model.transformer):
            name = entry[0]
            weight = entry[1]
            weights[f"transformer.{name}"] = weight

        mx.save_safetensors(str(path), weights)

    def _create_checkpoint_data(
        self, training_spec: TrainingSpec, start_date_time: datetime.datetime, files: dict
    ) -> dict:
        """Create checkpoint metadata dictionary.

        Args:
            training_spec: Training specification with mode and settings
            start_date_time: When training started
            files: Dictionary mapping component names to filenames in the checkpoint

        Returns:
            Dictionary with metadata (timing, mode, example count) and file manifest
        """
        now = datetime.datetime.now()
        return {
            "metadata": {
                "start": start_date_time.strftime("%Y-%m-%d %H:%M:%S"),
                "end": now.strftime("%Y-%m-%d %H:%M:%S"),
                "duration": self._format_duration(start_date_time, now),
                "number_of_training_examples": self.iterator.dataset.size(),
                "mode": training_spec.mode.value,
            },
            "files": files,
        }

    def should_save(self, training_spec: TrainingSpec) -> bool:
        return self.iterator.num_iterations % training_spec.saver.checkpoint_frequency == 0

    def should_plot_loss(self, training_spec: TrainingSpec) -> bool:
        if training_spec.instrumentation is None:
            return False
        return self.iterator.num_iterations % training_spec.instrumentation.plot_frequency == 0

    def should_generate_image(self, training_spec: TrainingSpec) -> bool:
        if training_spec.instrumentation is None:
            return False
        return self.iterator.num_iterations % training_spec.instrumentation.generate_image_frequency == 0

    def get_current_validation_image_path(self, training_spec: TrainingSpec) -> Path:
        output_path = Path(training_spec.saver.output_path) / TRAINING_PATH_VALIDATION_IMAGES
        output_path.mkdir(parents=True, exist_ok=True)
        path = output_path / Path(f"{self.iterator.num_iterations:07d}_{TRAINING_FILE_NAME_VALIDATION_IMAGE}.png")
        return path

    def get_current_loss_plot_path(self, training_spec: TrainingSpec) -> Path:
        output_path = Path(training_spec.saver.output_path) / TRAINING_PATH_VALIDATION_PLOT
        output_path.mkdir(parents=True, exist_ok=True)
        path = output_path / Path(f"{self.iterator.num_iterations:07d}_{TRAINING_FILE_NAME_VALIDATION_LOSS}.pdf")
        return path

    @staticmethod
    def _format_duration(start: datetime.datetime, end: datetime.datetime) -> str:
        """Format a time duration as human-readable string.

        Args:
            start: Start datetime
            end: End datetime

        Returns:
            Human-readable duration string (e.g., "2 hours and 30 minutes")
        """
        duration = end - start
        total_seconds = int(duration.total_seconds())

        hours, remainder = divmod(total_seconds, 3600)
        minutes, seconds = divmod(remainder, 60)

        parts = []
        if hours > 0:
            parts.append(f"{hours} hour{'s' if hours > 1 else ''}")
        if minutes > 0:
            parts.append(f"{minutes} minute{'s' if minutes > 1 else ''}")
        if seconds > 0 or not parts:
            parts.append(f"{seconds} second{'s' if seconds > 1 else ''}")

        return " and ".join(parts)

    @staticmethod
    def _save_train_config(path: Path, training_spec: TrainingSpec) -> None:
        """Save training configuration to checkpoint.

        Loads config from either the original config file path or from a previous
        checkpoint (for resumed training), then saves it to the new checkpoint.

        Args:
            path: Destination path for the config JSON file
            training_spec: Training specification containing config_path or checkpoint_path
        """
        if training_spec.config_path is not None:
            config_file_path = Path(training_spec.config_path)
            if not config_file_path.exists():
                logger.warning(f"Original config file not found: {config_file_path}. Skipping config save.")
                return
            with open(config_file_path, "r") as f:
                data = json.load(f)
                data["examples"]["path"] = str(config_file_path.parent / data["examples"]["path"])
        else:
            checkpoint = ZipUtil.unzip(
                training_spec.checkpoint_path, "checkpoint.json", lambda x: json.load(open(x, "r"))
            )
            data = ZipUtil.unzip(
                training_spec.checkpoint_path, checkpoint["files"]["config"], lambda x: json.load(open(x, "r"))
            )

        with open(path, "w", encoding="utf-8") as file:
            json.dump(data, file, indent=4)
