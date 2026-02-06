from __future__ import annotations

import datetime
import json
import random
from pathlib import Path
from typing import Any

from mflux.models.common.training.dataset.batch import Batch
from mflux.models.common.training.dataset.dataset import Dataset
from mflux.models.common.training.state.training_spec import TrainingSpec
from mflux.models.common.training.state.zip_util import ZipUtil


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
        self.total_items = dataset.size()
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
            return Iterator.from_json(training_spec=training_spec, iterator_path=training_spec.training_loop.iterator_state_path, dataset=dataset)  # fmt: off

        return Iterator(
            seed=training_spec.seed,
            dataset=dataset,
            batch_size=training_spec.training_loop.batch_size,
            num_epochs=training_spec.training_loop.num_epochs,
        )

    def _initialize_permutation(self):
        self._current_permutation = list(range(self.total_items))
        self.rng.shuffle(self._current_permutation)

    def __iter__(self):
        return self

    def __next__(self):
        if self.num_epochs is not None and self._epoch >= self.num_epochs:
            raise StopIteration()

        if self._position >= self.total_items:
            self._position = 0
            self._epoch += 1
            if self.num_epochs is not None and self._epoch >= self.num_epochs:
                raise StopIteration()
            self._initialize_permutation()

        remaining = self.total_items - self._position
        current_batch_size = min(self.batch_size, remaining)
        batch_indices = self._current_permutation[self._position : self._position + current_batch_size]
        self._position += current_batch_size
        self.num_iterations += 1

        data_items = [self.dataset.data[i] for i in batch_indices]
        return Batch(data=data_items, rng=self.rng)

    def to_dict(self) -> dict[str, Any]:
        return {
            "position": self._position,
            "epoch": self._epoch,
            "num_iterations": self.num_iterations,
            "current_permutation": self._current_permutation,
            "rng_state": list(self.rng.getstate()),
            "batch_size": self.batch_size,
            "num_epochs": self.num_epochs,
            "seed": self.seed,
            "start_date_time": self.start_date_time.strftime("%Y-%m-%d %H:%M:%S"),
        }

    @classmethod
    def from_dict(cls, state_dict: dict[str, Any], dataset: Dataset) -> "Iterator":
        rng_state = state_dict["rng_state"]
        rng_state = (rng_state[0], tuple(rng_state[1]), rng_state[2])
        return cls(
            dataset=dataset,
            batch_size=state_dict["batch_size"],
            num_epochs=state_dict["num_epochs"],
            seed=state_dict.get("seed"),
            position=state_dict["position"],
            epoch=state_dict["epoch"],
            num_iterations=state_dict["num_iterations"],
            current_permutation=state_dict["current_permutation"],
            rng_state=rng_state,
            start_date_time=datetime.datetime.strptime(state_dict["start_date_time"], "%Y-%m-%d %H:%M:%S"),
        )

    @classmethod
    def from_json(cls, training_spec: TrainingSpec, iterator_path: str, dataset: Dataset) -> "Iterator":
        data = ZipUtil.unzip(
            zip_path=training_spec.checkpoint_path,
            filename=iterator_path,
            loader=lambda x: json.load(open(x, "r")),
        )
        if "seed" not in data:
            data["seed"] = training_spec.seed
        return cls.from_dict(data, dataset)

    def save(self, path: Path) -> None:
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=4)

    def get_validation_batch(self) -> Batch:
        data_items = self.dataset.data
        total = len(data_items)
        if total > 10:
            # Deterministic sample without affecting training RNG.
            base_seed = self.seed or 0
            validation_rng = random.Random(base_seed + 1337 + self.num_iterations)
            indices = validation_rng.sample(range(total), k=10)
            data_items = [data_items[i] for i in indices]
        else:
            validation_rng = random.Random(self.seed or 0)
        return Batch(data=data_items, rng=validation_rng)

    def total_number_of_steps(self) -> int:
        total_epochs = self.num_epochs  # type: ignore[assignment]
        if total_epochs is None:
            return 0
        batches_per_epoch = (self.total_items + self.batch_size - 1) // self.batch_size
        return int(batches_per_epoch * total_epochs)
