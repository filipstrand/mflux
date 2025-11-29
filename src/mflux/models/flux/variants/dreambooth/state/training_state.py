import datetime
import json
import tempfile
import zipfile
from pathlib import Path
from typing import TYPE_CHECKING

from mflux.models.flux.variants.dreambooth.dataset.iterator import Iterator
from mflux.models.flux.variants.dreambooth.lora_layers.lora_layers import LoRALayers
from mflux.models.flux.variants.dreambooth.optimization.optimizer import Optimizer
from mflux.models.flux.variants.dreambooth.state.training_spec import TrainingSpec
from mflux.models.flux.variants.dreambooth.state.zip_util import ZipUtil
from mflux.models.flux.variants.dreambooth.statistics.statistics import Statistics

if TYPE_CHECKING:
    from mflux.models.flux.variants.txt2img.flux import Flux1

DREAMBOOTH_PATH_CHECKPOINTS = "_checkpoints"
DREAMBOOTH_FILE_NAME_CHECKPOINT = "checkpoint"

DREAMBOOTH_FILE_NAME_LORA_ADAPTER = "adapter"
DREAMBOOTH_FILE_NAME_OPTIMIZER = "optimizer"
DREAMBOOTH_FILE_NAME_ITERATOR = "iterator"
DREAMBOOTH_FILE_NAME_LOSS_FILE = "loss"
DREAMBOOTH_FILE_NAME_CONFIG_FILE = "config"

DREAMBOOTH_PATH_VALIDATION_IMAGES = "_validation/images/"
DREAMBOOTH_FILE_NAME_VALIDATION_IMAGE = "validation_image"

DREAMBOOTH_PATH_VALIDATION_PLOT = "_validation/plots/"
DREAMBOOTH_FILE_NAME_VALIDATION_LOSS = "loss"


class TrainingState:
    def __init__(
        self,
        iterator: Iterator,
        optimizer: Optimizer,
        statistics: Statistics,
    ):
        self.iterator = iterator
        self.optimizer = optimizer
        self.statistics = statistics

    def save(self, flux: "Flux1", training_spec: TrainingSpec) -> None:
        # Create a temporary directory to store files before zipping
        with tempfile.TemporaryDirectory() as temp_dir:
            # Save files to temporary directory
            optimizer_path = Path(temp_dir) / f"{self.iterator.num_iterations:07d}_{DREAMBOOTH_FILE_NAME_OPTIMIZER}.safetensors"  # fmt:off
            lora_path = Path(temp_dir) / f"{self.iterator.num_iterations:07d}_{DREAMBOOTH_FILE_NAME_LORA_ADAPTER}.safetensors"  # fmt:off
            iterator_path = Path(temp_dir) / f"{self.iterator.num_iterations:07d}_{DREAMBOOTH_FILE_NAME_ITERATOR}.json"
            loss_path = Path(temp_dir) / f"{self.iterator.num_iterations:07d}_{DREAMBOOTH_FILE_NAME_LOSS_FILE}.json"
            config_path = Path(temp_dir) / f"{self.iterator.num_iterations:07d}_{DREAMBOOTH_FILE_NAME_CONFIG_FILE}.json"
            checkpoint_path = Path(temp_dir) / f"{DREAMBOOTH_FILE_NAME_CHECKPOINT}.json"
            paths = [optimizer_path, lora_path, iterator_path, loss_path, config_path, checkpoint_path]

            # Save individual files to temporary directory
            self.optimizer.save(optimizer_path)
            LoRALayers.save(flux.transformer, lora_path, training_spec)
            self.iterator.save(iterator_path)
            self.statistics.save(loss_path)
            self._save_train_config(config_path, training_spec)  # For completeness, we also save the (initial) config

            # Create checkpoint data
            checkpoint_data = self._create_checkpoint_data(training_spec, self.iterator.start_date_time)
            with open(checkpoint_path, "w") as json_file:
                json.dump(checkpoint_data, json_file, indent=4)

            # Create zip file
            output_path = Path(training_spec.saver.output_path) / DREAMBOOTH_PATH_CHECKPOINTS
            output_path.mkdir(parents=True, exist_ok=True)
            zip_path = output_path / f"{self.iterator.num_iterations:07d}_checkpoint.zip"

            with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
                for file_path in paths:
                    if file_path.exists():
                        zipf.write(file_path, file_path.name)

    def _create_checkpoint_data(self, training_spec: TrainingSpec, start_date_time: datetime.datetime) -> dict:
        now = datetime.datetime.now()
        return {
            "metadata": {
                "start": start_date_time.strftime("%Y-%m-%d %H:%M:%S"),
                "end": now.strftime("%Y-%m-%d %H:%M:%S"),
                "duration": self._format_duration(start_date_time, now),
                "number_of_training_examples": self.iterator.dataset.size(),
            },
            "files": {
                "config": f"{self.iterator.num_iterations:07d}_{DREAMBOOTH_FILE_NAME_CONFIG_FILE}.json",
                "optimizer": f"{self.iterator.num_iterations:07d}_{DREAMBOOTH_FILE_NAME_OPTIMIZER}.safetensors",
                "lora_adapter": f"{self.iterator.num_iterations:07d}_{DREAMBOOTH_FILE_NAME_LORA_ADAPTER}.safetensors",
                "iterator": f"{self.iterator.num_iterations:07d}_{DREAMBOOTH_FILE_NAME_ITERATOR}.json",
                "loss": f"{self.iterator.num_iterations:07d}_{DREAMBOOTH_FILE_NAME_LOSS_FILE}.json",
            },
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
        output_path = Path(training_spec.saver.output_path) / DREAMBOOTH_PATH_VALIDATION_IMAGES
        output_path.mkdir(parents=True, exist_ok=True)
        path = output_path / Path(f"{self.iterator.num_iterations:07d}_{DREAMBOOTH_FILE_NAME_VALIDATION_IMAGE}.png")
        return path

    def get_current_loss_plot_path(self, training_spec: TrainingSpec) -> Path:
        output_path = Path(training_spec.saver.output_path) / DREAMBOOTH_PATH_VALIDATION_PLOT
        output_path.mkdir(parents=True, exist_ok=True)
        path = output_path / Path(f"{self.iterator.num_iterations:07d}_{DREAMBOOTH_FILE_NAME_VALIDATION_LOSS}.pdf")
        return path

    @staticmethod
    def _get_parent_path(path: str | None) -> str | None:
        return None if path is None else str(Path(path).resolve())

    @staticmethod
    def _format_duration(start: datetime.datetime, end: datetime.datetime) -> str:
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

    @staticmethod
    def _save_train_config(path: Path, training_spec: TrainingSpec) -> None:
        if training_spec.config_path is not None:
            with open(Path(training_spec.config_path), "r") as f:
                data = json.load(f)
                # Since the zip file can be moved to an arbitrary location, we do a slight
                # modification and override the image location with the absolute path of the image files
                data["examples"]["path"] = str(Path(training_spec.config_path).parent / data["examples"]["path"])
        else:
            checkpoint = ZipUtil.unzip(
                training_spec.checkpoint_path, "checkpoint.json", lambda x: json.load(open(x, "r"))
            )
            data = ZipUtil.unzip(
                training_spec.checkpoint_path, checkpoint["files"]["config"], lambda x: json.load(open(x, "r"))
            )

        with open(path, "w", encoding="utf-8") as file:
            json.dump(data, file, indent=4)
