from __future__ import annotations

import datetime
import json
from pathlib import Path

from mflux.models.common.training.state.training_spec import TrainingSpec
from mflux.models.common.training.state.zip_util import ZipUtil


class Statistics:
    def __init__(self):
        self.steps: list[int] = []
        self.losses: list[float] = []
        self.times: list[datetime.datetime] = []

    @staticmethod
    def from_spec(training_spec: TrainingSpec) -> "Statistics":
        if training_spec.statistics is None or training_spec.statistics.state_path is None:
            return Statistics()

        stats = Statistics()
        data = ZipUtil.unzip(
            zip_path=training_spec.checkpoint_path,
            filename=training_spec.statistics.state_path,
            loader=lambda x: json.load(open(x, "r")),
        )
        for entry in data:
            stats.steps.append(entry["step"])
            stats.losses.append(entry["loss"])
            stats.times.append(datetime.datetime.strptime(entry["time"], "%Y-%m-%d %H:%M:%S"))

        return stats

    def append_values(self, step: int, loss: float) -> None:
        self.steps.append(step)
        self.losses.append(loss)
        self.times.append(datetime.datetime.now())

    def save(self, path: Path) -> None:
        loss_entries = [
            {"step": step, "loss": float(loss), "time": time.strftime("%Y-%m-%d %H:%M:%S")}
            for step, loss, time in zip(self.steps, self.losses, self.times)
        ]
        with open(path, "w", encoding="utf-8") as file:
            json.dump(loss_entries, file, indent=4)
