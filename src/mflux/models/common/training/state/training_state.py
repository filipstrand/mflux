from __future__ import annotations

import datetime
import json
import os
import tempfile
import zipfile
from pathlib import Path
from typing import Protocol

from mflux.models.common.training.dataset.iterator import Iterator
from mflux.models.common.training.optimization.optimizer import Optimizer
from mflux.models.common.training.state.training_spec import (
    TRAINING_FILE_NAME_RUN_MANIFEST,
    TrainingSpec,
)
from mflux.models.common.training.state.zip_util import ZipUtil
from mflux.models.common.training.statistics.statistics import Statistics


class TrainingAdapterForCheckpointing(Protocol):
    def save_lora_adapter(self, *, path: Path, training_spec: TrainingSpec) -> None: ...


TRAINING_PATH_CHECKPOINTS = "checkpoints"
TRAINING_FILE_NAME_CHECKPOINT = "checkpoint"

TRAINING_FILE_NAME_LORA_ADAPTER = "adapter"
TRAINING_FILE_NAME_OPTIMIZER = "optimizer"
TRAINING_FILE_NAME_ITERATOR = "iterator"
TRAINING_FILE_NAME_LOSS_FILE = "loss"
TRAINING_FILE_NAME_CONFIG_FILE = "config"

TRAINING_PATH_PREVIEW_IMAGES = "preview/"
TRAINING_FILE_NAME_PREVIEW_IMAGE = "preview_image"

TRAINING_PATH_VALIDATION_PLOT = "loss/"
TRAINING_FILE_NAME_VALIDATION_LOSS = "loss"


class TrainingState:
    def __init__(self, iterator: Iterator, optimizer: Optimizer, statistics: Statistics):
        self.iterator = iterator
        self.optimizer = optimizer
        self.statistics = statistics

    def save(self, adapter: TrainingAdapterForCheckpointing, training_spec: TrainingSpec) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            optimizer_path = Path(temp_dir) / f"{self.iterator.num_iterations:07d}_{TRAINING_FILE_NAME_OPTIMIZER}.safetensors"  # fmt: off
            lora_path = Path(temp_dir) / f"{self.iterator.num_iterations:07d}_{TRAINING_FILE_NAME_LORA_ADAPTER}.safetensors"  # fmt: off
            iterator_path = Path(temp_dir) / f"{self.iterator.num_iterations:07d}_{TRAINING_FILE_NAME_ITERATOR}.json"
            loss_path = Path(temp_dir) / f"{self.iterator.num_iterations:07d}_{TRAINING_FILE_NAME_LOSS_FILE}.json"
            config_path = Path(temp_dir) / f"{self.iterator.num_iterations:07d}_{TRAINING_FILE_NAME_CONFIG_FILE}.json"
            checkpoint_path = Path(temp_dir) / f"{TRAINING_FILE_NAME_CHECKPOINT}.json"
            manifest_path = Path(temp_dir) / TRAINING_FILE_NAME_RUN_MANIFEST
            paths = [optimizer_path, lora_path, iterator_path, loss_path, config_path, checkpoint_path, manifest_path]

            self.optimizer.save(optimizer_path)
            adapter.save_lora_adapter(path=lora_path, training_spec=training_spec)
            self.iterator.save(iterator_path)
            self.statistics.save(loss_path)
            self._save_train_config(config_path, training_spec)

            checkpoint_data = self._create_checkpoint_data(training_spec, self.iterator.start_date_time)
            with open(checkpoint_path, "w") as json_file:
                json.dump(checkpoint_data, json_file, indent=4)

            output_path = Path(training_spec.checkpoint.output_path) / TRAINING_PATH_CHECKPOINTS
            output_path.mkdir(parents=True, exist_ok=True)
            zip_path = output_path / f"{self.iterator.num_iterations:07d}_checkpoint.zip"

            run_manifest = self._create_run_manifest(
                training_spec=training_spec,
                checkpoint_data=checkpoint_data,
                checkpoint_dir=output_path,
            )
            with open(manifest_path, "w", encoding="utf-8") as json_file:
                json.dump(run_manifest, json_file, indent=4)

            with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
                for file_path in paths:
                    if file_path.exists():
                        zipf.write(file_path, file_path.name)

    def _create_run_manifest(
        self,
        *,
        training_spec: TrainingSpec,
        checkpoint_data: dict,
        checkpoint_dir: Path,
    ) -> dict:
        data_root = training_spec.data_root
        relative_data_root = None
        if data_root:
            try:
                relative_data_root = os.path.relpath(data_root, checkpoint_dir)
            except ValueError:
                relative_data_root = None

        return {
            "version": 1,
            "created_at": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "model": training_spec.model,
            "is_edit": training_spec.is_edit,
            "data_root": data_root,
            "data_root_relative": relative_data_root,
            "data_fingerprint": self._create_data_fingerprint(training_spec),
            "checkpoint_files": checkpoint_data["files"],
        }

    @staticmethod
    def _create_data_fingerprint(training_spec: TrainingSpec) -> dict:
        data_root = training_spec.data_root
        images: list[str] = []
        input_images: list[str] = []
        if data_root:
            root_path = Path(data_root).resolve()
            for item in training_spec.data:
                resolved_path = item.image.resolve()
                if resolved_path.is_relative_to(root_path):
                    images.append(str(resolved_path.relative_to(root_path)))
                else:
                    images.append(resolved_path.name)
                if item.input_image is not None:
                    resolved_input = item.input_image.resolve()
                    if resolved_input.is_relative_to(root_path):
                        input_images.append(str(resolved_input.relative_to(root_path)))
                    else:
                        input_images.append(resolved_input.name)

        return {
            "count": len(training_spec.data),
            "images": sorted(images),
            "input_images": sorted(input_images),
            "is_edit": training_spec.is_edit,
        }

    def _create_checkpoint_data(self, training_spec: TrainingSpec, start_date_time: datetime.datetime) -> dict:
        now = datetime.datetime.now()
        return {
            "metadata": {
                "start": start_date_time.strftime("%Y-%m-%d %H:%M:%S"),
                "end": now.strftime("%Y-%m-%d %H:%M:%S"),
                "duration": self._format_duration(start_date_time, now),
                "number_of_training_data": self.iterator.dataset.size(),
            },
            "files": {
                "config": f"{self.iterator.num_iterations:07d}_{TRAINING_FILE_NAME_CONFIG_FILE}.json",
                "optimizer": f"{self.iterator.num_iterations:07d}_{TRAINING_FILE_NAME_OPTIMIZER}.safetensors",
                "lora_adapter": f"{self.iterator.num_iterations:07d}_{TRAINING_FILE_NAME_LORA_ADAPTER}.safetensors",
                "iterator": f"{self.iterator.num_iterations:07d}_{TRAINING_FILE_NAME_ITERATOR}.json",
                "loss": f"{self.iterator.num_iterations:07d}_{TRAINING_FILE_NAME_LOSS_FILE}.json",
            },
        }

    def should_save(self, training_spec: TrainingSpec) -> bool:
        return self.iterator.num_iterations % training_spec.checkpoint.save_frequency == 0

    def should_plot_loss(self, training_spec: TrainingSpec) -> bool:
        if training_spec.monitoring is None:
            return False
        return self.iterator.num_iterations % training_spec.monitoring.plot_frequency == 0

    def should_generate_image(self, training_spec: TrainingSpec) -> bool:
        if training_spec.monitoring is None:
            return False
        return self.iterator.num_iterations % training_spec.monitoring.generate_image_frequency == 0

    def get_current_preview_image_path(
        self,
        training_spec: TrainingSpec,
        *,
        preview_index: int,
        preview_name: str | None = None,
    ) -> Path:
        output_path = Path(training_spec.checkpoint.output_path) / TRAINING_PATH_PREVIEW_IMAGES
        output_path.mkdir(parents=True, exist_ok=True)
        suffix = preview_name or f"{preview_index + 1:02d}"
        return output_path / Path(f"{self.iterator.num_iterations:07d}_{TRAINING_FILE_NAME_PREVIEW_IMAGE}_{suffix}.png")

    def get_current_loss_plot_path(self, training_spec: TrainingSpec) -> Path:
        output_path = Path(training_spec.checkpoint.output_path) / TRAINING_PATH_VALIDATION_PLOT
        output_path.mkdir(parents=True, exist_ok=True)
        return output_path / Path(f"{TRAINING_FILE_NAME_VALIDATION_LOSS}.html")

    @staticmethod
    def _format_duration(start: datetime.datetime, end: datetime.datetime) -> str:
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
        if training_spec.config_path is not None:
            with open(Path(training_spec.config_path), "r") as f:
                data = json.load(f)
                # Make data path absolute inside checkpoint config.
                data_conf = data["data"]
                if isinstance(data_conf, str):
                    resolved_path = Path(training_spec.config_path).parent / data_conf
                elif isinstance(data_conf, dict) and "path" in data_conf:
                    resolved_path = Path(training_spec.config_path).parent / str(data_conf["path"])
                else:
                    raise ValueError("Config 'data' must be a string or {'path': ...}.")
                data["data"] = str(resolved_path)
        else:
            checkpoint = ZipUtil.unzip(
                training_spec.checkpoint_path, "checkpoint.json", lambda x: json.load(open(x, "r"))
            )
            data = ZipUtil.unzip(
                training_spec.checkpoint_path, checkpoint["files"]["config"], lambda x: json.load(open(x, "r"))
            )
            if training_spec.data_root is not None:
                data["data"] = training_spec.data_root
        with open(path, "w", encoding="utf-8") as file:
            json.dump(data, file, indent=4)
