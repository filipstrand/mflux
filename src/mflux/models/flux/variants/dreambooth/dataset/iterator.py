import datetime
import json
import random
from pathlib import Path
from typing import Any

from mflux.models.flux.variants.dreambooth.dataset.batch import Batch
from mflux.models.flux.variants.dreambooth.dataset.dataset import Dataset
from mflux.models.flux.variants.dreambooth.state.training_spec import TrainingSpec
from mflux.models.flux.variants.dreambooth.state.zip_util import ZipUtil


class Iterator:
    def __init__(
        self,
        dataset: Dataset,
        batch_size: int,
        num_epochs: int | None = None,
        seed: int | None = None,
        position: int = 0,
        epoch: int = 0,
        num_iterations: int = 0,
        current_permutation: list[int] | None = None,
        rng_state: tuple | None = None,
        start_date_time: datetime.datetime | None = None,
    ):
        self.dataset = dataset
        self.batch_size = batch_size
        self.total_examples = dataset.size()
        self.num_epochs = num_epochs
        self.seed = seed
        self.rng = random.Random(seed)

        self._position = position
        self._epoch = epoch
        self.num_iterations = num_iterations
        self.start_date_time = start_date_time or datetime.datetime.now()

        if rng_state is not None:
            self.rng.setstate(rng_state)

        if current_permutation is not None:
            self._current_permutation = current_permutation
        else:
            self._initialize_permutation()

    @staticmethod
    def from_spec(training_spec: TrainingSpec, dataset: Dataset) -> "Iterator":
        if training_spec.training_loop.iterator_state_path is not None:
            return Iterator.from_json(
                training_spec=training_spec,
                iterator_path=training_spec.training_loop.iterator_state_path,
                dataset=dataset,
            )

        return Iterator(
            seed=training_spec.seed,
            dataset=dataset,
            batch_size=training_spec.training_loop.batch_size,
            num_epochs=training_spec.training_loop.num_epochs,
        )

    def _initialize_permutation(self):
        self._current_permutation = list(range(self.total_examples))
        self.rng.shuffle(self._current_permutation)

    def __iter__(self):
        return self

    def __next__(self):
        # Check if we've reached the specified number of epochs
        if self.num_epochs is not None and self._epoch >= self.num_epochs:
            raise StopIteration()

        # If we've used all examples, start new epoch
        if self._position >= self.total_examples:
            self._position = 0
            self._epoch += 1

            # Check again after incrementing epoch
            if self.num_epochs is not None and self._epoch >= self.num_epochs:
                raise StopIteration()

            self._initialize_permutation()

        # Calculate batch size for this iteration
        remaining = self.total_examples - self._position
        current_batch_size = min(self.batch_size, remaining)

        # Get indices for current batch
        batch_indices = self._current_permutation[self._position : self._position + current_batch_size]

        # Update position
        self._position += current_batch_size

        # Increment iteration counter
        self.num_iterations += 1

        # Get examples using indices
        examples = [self.dataset.examples[i] for i in batch_indices]

        return Batch(examples=examples, rng=self.rng)

    def to_dict(self) -> dict[str, Any]:
        return {
            "position": self._position,
            "epoch": self._epoch,
            "num_iterations": self.num_iterations,
            "current_permutation": self._current_permutation,
            "rng_state": list(self.rng.getstate()),
            "batch_size": self.batch_size,
            "num_epochs": self.num_epochs,
            "start_date_time": self.start_date_time.strftime("%Y-%m-%d %H:%M:%S"),
        }

    @classmethod
    def from_dict(cls, state_dict: dict[str, Any], dataset: Dataset) -> "Iterator":
        # Convert the RNG state elements to tuples
        rng_state = state_dict["rng_state"]
        rng_state = (rng_state[0], tuple(rng_state[1]), rng_state[2])

        return cls(
            dataset=dataset,
            batch_size=state_dict["batch_size"],
            num_epochs=state_dict["num_epochs"],
            position=state_dict["position"],
            epoch=state_dict["epoch"],
            num_iterations=state_dict["num_iterations"],
            current_permutation=state_dict["current_permutation"],
            rng_state=rng_state,
            start_date_time=datetime.datetime.strptime(state_dict["start_date_time"], "%Y-%m-%d %H:%M:%S"),
        )

    def to_json(self, filepath: str) -> None:
        with open(filepath, "w") as f:
            json.dump(self.to_dict(), f, indent=4)

    @classmethod
    def from_json(cls, training_spec: TrainingSpec, iterator_path: str, dataset: Dataset) -> "Iterator":
        data = ZipUtil.unzip(
            zip_path=training_spec.checkpoint_path,
            filename=iterator_path,
            loader=lambda x: json.load(open(x, "r")),
        )
        return cls.from_dict(data, dataset)

    def get_validation_batch(self) -> Batch:
        # This is of course a misleading label, and future updates should include actual validation image inputs.
        examples = self.dataset.examples
        if len(examples) > 10:
            examples = examples[0:10]

        return Batch(examples=examples, rng=self.rng)

    def total_number_of_steps(self) -> int:
        return self.total_examples * self.num_epochs  # type: ignore

    def save(self, path: Path) -> None:
        self.to_json(str(path))
