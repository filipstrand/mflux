import json
from pathlib import Path

from mflux.models.z_image.variants.training.state.training_spec import TrainingSpec
from mflux.models.z_image.variants.training.state.zip_util import ZipUtil


class Statistics:
    """Training statistics tracking for loss visualization."""

    def __init__(self, steps: list[int] | None = None, losses: list[float] | None = None):
        self.steps = steps or []
        self.losses = losses or []

    @staticmethod
    def from_spec(training_spec: TrainingSpec) -> "Statistics":
        """Load statistics from checkpoint or create new."""
        if training_spec.statistics.state_path is not None:
            return ZipUtil.unzip(
                zip_path=training_spec.checkpoint_path,
                filename=training_spec.statistics.state_path,
                loader=lambda path: Statistics.from_json(path),
            )
        return Statistics()

    def append_values(self, step: int, loss: float) -> None:
        """Record a loss value at a training step."""
        self.steps.append(step)
        self.losses.append(float(loss))

    def to_dict(self) -> dict:
        return {
            "steps": self.steps,
            "losses": self.losses,
        }

    @staticmethod
    def from_dict(data: dict) -> "Statistics":
        return Statistics(
            steps=data.get("steps", []),
            losses=data.get("losses", []),
        )

    def to_json(self, path: Path) -> None:
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=4)

    @staticmethod
    def from_json(path: str) -> "Statistics":
        with open(path, "r") as f:
            data = json.load(f)
        return Statistics.from_dict(data)

    def save(self, path: Path) -> None:
        self.to_json(path)
