import datetime
import json
from pathlib import Path

from mflux.dreambooth.dataset.iterator import Iterator
from mflux.dreambooth.lora_layers.lora_layers import LoRALayers
from mflux.dreambooth.optimization.optimizer import Optimizer
from mflux.dreambooth.state.training_spec import TrainingSpec
from mflux.dreambooth.statistics.statistics import Statistics

DREAMBOOTH_PATH_CHECKPOINTS = "_checkpoints"
DREAMBOOTH_FILE_NAME_CHECKPOINT = "checkpoint"

DREAMBOOTH_PATH_LORA_ADAPTERS = "_state/lora_adapters"
DREAMBOOTH_FILE_NAME_LORA_ADAPTER = "adapter"

DREAMBOOTH_PATH_OPTIMIZATION = "_state/optimization"
DREAMBOOTH_FILE_NAME_OPTIMIZER = "optimizer"

DREAMBOOTH_PATH_ITERATOR = "_state/iterator"
DREAMBOOTH_FILE_NAME_ITERATOR = "iterator"

DREAMBOOTH_PATH_STATISTICS = "_state/statistics/"
DREAMBOOTH_FILE_NAME_STATISTICS = "statistics"
DREAMBOOTH_FILE_NAME_LOSS_FILE = "loss"

DREAMBOOTH_PATH_VALIDATION_IMAGES = "_validation/images/"
DREAMBOOTH_FILE_NAME_VALIDATION_IMAGE = "validation_image"

DREAMBOOTH_PATH_VALIDATION_PLOT = "_validation/plots/"
DREAMBOOTH_FILE_NAME_VALIDATION_LOSS = "loss"

DREAMBOOTH_PATH_TRAIN_SPECIFICATION = "training_specification"
DREAMBOOTH_FILE_NAME_SPECIFICATION = "specification"


class TrainingState:
    def __init__(
        self,
        iterator: Iterator,
        lora_layers: LoRALayers,
        optimizer: Optimizer,
        statistics: Statistics,
    ):
        self.iterator = iterator
        self.optimizer = optimizer
        self.lora_layers = lora_layers
        self.statistics = statistics

    def save(self, training_spec: TrainingSpec) -> None:
        self.optimizer.save(self.get_current_optimizer_path(training_spec))
        self.lora_layers.save(self.get_current_lora_layers_path(training_spec), training_spec)
        self.iterator.save(self.get_current_iterator_path(training_spec))
        self.save_checkpoint_file(training_spec, self.iterator.start_date_time)

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
        output_path = Path(training_spec.saver.output_path) / DREAMBOOTH_PATH_VALIDATION_IMAGES
        output_path.mkdir(parents=True, exist_ok=True)
        path = output_path / Path(f"{self.iterator.num_iterations :07d}_{DREAMBOOTH_FILE_NAME_VALIDATION_IMAGE}.png")
        return path

    def get_current_lora_layers_path(self, training_spec: TrainingSpec) -> Path:
        lora_path = Path(training_spec.saver.output_path) / DREAMBOOTH_PATH_LORA_ADAPTERS
        lora_path.mkdir(parents=True, exist_ok=True)
        path = lora_path / f"{self.iterator.num_iterations :07d}_{DREAMBOOTH_FILE_NAME_LORA_ADAPTER}.safetensors"
        return path

    def get_current_iterator_path(self, training_spec: TrainingSpec) -> Path:
        lora_path = Path(training_spec.saver.output_path) / DREAMBOOTH_PATH_ITERATOR
        lora_path.mkdir(parents=True, exist_ok=True)
        path = lora_path / f"{self.iterator.num_iterations :07d}_{DREAMBOOTH_FILE_NAME_ITERATOR}.json"
        return path

    def get_current_optimizer_path(self, training_spec: TrainingSpec) -> Path:
        optimizer_path = Path(training_spec.saver.output_path) / DREAMBOOTH_PATH_OPTIMIZATION
        optimizer_path.mkdir(parents=True, exist_ok=True)
        path = optimizer_path / f"{self.iterator.num_iterations :07d}_{DREAMBOOTH_FILE_NAME_OPTIMIZER}.safetensors"
        return path

    def get_current_checkpoint_path(self, training_spec: TrainingSpec) -> Path:
        output_path = Path(training_spec.saver.output_path) / DREAMBOOTH_PATH_CHECKPOINTS
        output_path.mkdir(parents=True, exist_ok=True)
        path = output_path / Path(f"{self.iterator.num_iterations :07d}_{DREAMBOOTH_FILE_NAME_CHECKPOINT}.json")
        return path

    def get_current_loss_plot_path(self, training_spec: TrainingSpec) -> Path:
        output_path = Path(training_spec.saver.output_path) / DREAMBOOTH_PATH_VALIDATION_PLOT
        output_path.mkdir(parents=True, exist_ok=True)
        path = output_path / Path(f"{self.iterator.num_iterations :07d}_{DREAMBOOTH_FILE_NAME_VALIDATION_LOSS}.pdf")
        return path

    def get_loss_file_path(self, training_spec: TrainingSpec) -> Path:
        output_path = Path(training_spec.saver.output_path) / DREAMBOOTH_PATH_STATISTICS
        output_path.mkdir(parents=True, exist_ok=True)
        path = output_path / Path(f"{self.iterator.num_iterations :07d}_{DREAMBOOTH_FILE_NAME_LOSS_FILE}.json")
        return path

    @staticmethod
    def get_train_spec_file_path(training_spec: TrainingSpec) -> Path:
        output_path = Path(training_spec.saver.output_path) / DREAMBOOTH_PATH_TRAIN_SPECIFICATION
        output_path.mkdir(parents=True, exist_ok=True)
        path = output_path / Path(f"{DREAMBOOTH_FILE_NAME_SPECIFICATION}.json")
        return path

    def save_training_spec_copy(self, path: Path, training_spec: TrainingSpec) -> None:
        with open(self.get_train_spec_file_path(training_spec), "w") as json_file:
            json_file.write(training_spec.to_json())

    def save_checkpoint_file(self, training_spec: TrainingSpec, start_date_time: datetime) -> None:
        now = datetime.datetime.now()
        ckpt = {
            "config": TrainingState._get_parent_path(training_spec.config_path),
            DREAMBOOTH_FILE_NAME_LORA_ADAPTER: str(self.get_current_lora_layers_path(training_spec)),
            DREAMBOOTH_FILE_NAME_OPTIMIZER: str(self.get_current_optimizer_path(training_spec)),
            DREAMBOOTH_FILE_NAME_ITERATOR: str(self.get_current_iterator_path(training_spec)),
            DREAMBOOTH_FILE_NAME_STATISTICS: str(self.get_loss_file_path(training_spec)),
            "metadata": {
                "start": start_date_time.strftime("%Y-%m-%d %H:%M:%S"),
                "end": now.strftime("%Y-%m-%d %H:%M:%S"),
                "duration": TrainingState._format_duration(start_date_time, now),
                "number_of_training_examples": self.iterator.dataset.size(),
            },
        }
        with open(self.get_current_checkpoint_path(training_spec), "w") as json_file:
            json.dump(ckpt, json_file, indent=4)

    @staticmethod
    def _get_parent_path(path: str | None) -> str | None:
        return None if path is None else str(Path(path).resolve())

    @staticmethod
    def _format_duration(start: datetime, end: datetime) -> str:
        # Calculate the duration
        duration = end - start
        total_seconds = int(duration.total_seconds())

        # Calculate hours, minutes, and seconds
        hours, remainder = divmod(total_seconds, 3600)
        minutes, seconds = divmod(remainder, 60)

        # Build the string
        parts = []
        if hours > 0:
            parts.append(f"{hours} hour{'s' if hours > 1 else ''}")
        if minutes > 0:
            parts.append(f"{minutes} minute{'s' if minutes > 1 else ''}")
        if seconds > 0 or not parts:  # Include seconds even if zero if no other parts exist
            parts.append(f"{seconds} second{'s' if seconds > 1 else ''}")

        return " and ".join(parts)
