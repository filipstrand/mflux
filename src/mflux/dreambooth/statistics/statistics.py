import datetime
import json
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mflux.dreambooth.state.training_spec import TrainingSpec
    from mflux.dreambooth.state.training_state import TrainingState


class Statistics:
    def __init__(self):
        self.steps: list[int] = []
        self.losses: list[float] = []
        self.times: list[datetime.datetime] = []

    @staticmethod
    def from_spec(training_spec: "TrainingSpec") -> "Statistics":
        if training_spec.statistics is None:
            return Statistics()

        if training_spec.statistics.state_path is None:
            return Statistics()

        with open(training_spec.statistics.state_path, "r") as f:
            data = json.load(f)

        stats = Statistics()

        for entry in data:
            stats.steps.append(entry["step"])
            stats.losses.append(entry["loss"])
            stats.times.append(datetime.datetime.strptime(entry["time"], "%Y-%m-%d %H:%M:%S"))

        return stats

    def append_values(self, step: int, loss: float) -> None:
        self.steps.append(step)
        self.losses.append(loss)
        self.times.append(datetime.datetime.now())

    def update_loss_file(self, training_spec: "TrainingSpec", training_state: "TrainingState") -> None:
        loss_entries = [
            {
                "step": step,
                "loss": float(loss),
                "time": time.strftime("%Y-%m-%d %H:%M:%S")
            }
            for step, loss, time in zip(self.steps, self.losses, self.times)
        ]  # fmt: off

        with open(training_state.get_loss_file_path(training_spec), "w", encoding="utf-8") as file:
            json.dump(loss_entries, file, indent=4)
