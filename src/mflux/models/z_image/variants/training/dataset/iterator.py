import datetime
import json
import logging
import random
from pathlib import Path
from typing import TYPE_CHECKING, Any

from mflux.models.z_image.variants.training.dataset.batch import Batch
from mflux.models.z_image.variants.training.dataset.dataset import Dataset
from mflux.models.z_image.variants.training.state.training_spec import TrainingSpec
from mflux.models.z_image.variants.training.state.zip_util import ZipUtil

logger = logging.getLogger(__name__)

# Maximum number of examples to use for validation loss computation.
# Kept small (10) to avoid memory pressure during training validation.
# For systems with more memory (512GB+), this could be increased to 50-100
# for more representative validation metrics at the cost of longer validation time.
VALIDATION_BATCH_MAX_SIZE = 10

if TYPE_CHECKING:
    from mflux.models.z_image.variants.training.dataset.aspect_ratio_sampler import AspectRatioSampler


class Iterator:
    """Dataset iterator with state persistence for training resumption.

    Supports two sampling modes:
    - Default: Simple shuffled permutation of all examples
    - Aspect ratio bucketing: Groups examples by aspect ratio for efficient batching
    """

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
        use_aspect_ratio_bucketing: bool = False,
    ):
        self.dataset = dataset
        self.batch_size = batch_size
        self.total_examples = dataset.size()
        self.num_epochs = num_epochs
        self.seed = seed
        self.rng = random.Random(seed)
        self.use_aspect_ratio_bucketing = use_aspect_ratio_bucketing

        self._position = position
        self._epoch = epoch
        self.num_iterations = num_iterations
        self.start_date_time = start_date_time or datetime.datetime.now()

        # Aspect ratio sampler (lazy initialized)
        self._aspect_ratio_sampler: "AspectRatioSampler | None" = None
        self._current_target_resolution: tuple[int, int] | None = None

        if rng_state is not None:
            # Validate RNG state structure before applying
            if not isinstance(rng_state, tuple) or len(rng_state) != 3:
                raise ValueError(f"Invalid rng_state: expected tuple of 3 elements, got {type(rng_state)}")
            if not isinstance(rng_state[1], tuple):
                raise ValueError(f"Invalid rng_state[1]: expected tuple, got {type(rng_state[1])}")
            self.rng.setstate(rng_state)

        if current_permutation is not None:
            self._current_permutation = current_permutation
        else:
            self._initialize_permutation()

        # Initialize aspect ratio sampler if enabled
        if use_aspect_ratio_bucketing:
            self._init_aspect_ratio_sampler()

    @staticmethod
    def from_spec(training_spec: TrainingSpec, dataset: Dataset) -> "Iterator":
        if training_spec.training_loop.iterator_state_path is not None:
            return Iterator.from_json(
                training_spec=training_spec,
                iterator_path=training_spec.training_loop.iterator_state_path,
                dataset=dataset,
            )

        # Check if aspect ratio bucketing is enabled in dataset config
        use_bucketing = False
        if training_spec.dataset is not None:
            use_bucketing = getattr(training_spec.dataset, "use_aspect_ratio_bucketing", False)

        return Iterator(
            seed=training_spec.seed,
            dataset=dataset,
            batch_size=training_spec.training_loop.batch_size,
            num_epochs=training_spec.training_loop.num_epochs,
            use_aspect_ratio_bucketing=use_bucketing,
        )

    def _init_aspect_ratio_sampler(self) -> None:
        """Initialize the aspect ratio sampler."""
        from mflux.models.z_image.variants.training.dataset.aspect_ratio_sampler import (
            AspectRatioSampler,
        )

        self._aspect_ratio_sampler = AspectRatioSampler(
            examples=self.dataset.examples,
            seed=self.seed,
        )

    @property
    def current_target_resolution(self) -> tuple[int, int] | None:
        """Get the target resolution for the current batch (for aspect ratio bucketing)."""
        return self._current_target_resolution

    def _initialize_permutation(self):
        self._current_permutation = list(range(self.total_examples))
        self.rng.shuffle(self._current_permutation)

    def __iter__(self):
        return self

    def __next__(self):
        # Use aspect ratio bucketing if enabled
        if self.use_aspect_ratio_bucketing:
            if self._aspect_ratio_sampler is None:
                logger.warning(
                    "Aspect ratio bucketing enabled but sampler not initialized, falling back to standard iteration"
                )
                self.use_aspect_ratio_bucketing = False  # Disable to avoid repeated warnings
            else:
                return self._next_with_bucketing()
        return self._next_standard()

    def _next_standard(self) -> Batch:
        """Standard iteration without aspect ratio bucketing."""
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

        # Clear target resolution (not used in standard mode)
        self._current_target_resolution = None

        # Get examples using indices
        examples = [self.dataset.examples[i] for i in batch_indices]

        return Batch(examples=examples, rng=self.rng)

    def _next_with_bucketing(self) -> Batch:
        """Iteration with aspect ratio bucketing."""
        # Check if we've reached the specified number of epochs
        if self.num_epochs is not None and self._epoch >= self.num_epochs:
            raise StopIteration()

        # Safety counter to prevent infinite loops
        max_reset_attempts = 3

        # Get batch from aspect ratio sampler
        result = self._aspect_ratio_sampler.get_batch(self.batch_size)

        reset_attempts = 0
        while result is None and reset_attempts < max_reset_attempts:
            # No more examples in current epoch, start new epoch
            self._epoch += 1
            self._position = 0

            # Check if we've exceeded epochs
            if self.num_epochs is not None and self._epoch >= self.num_epochs:
                raise StopIteration()

            # Reset sampler for new epoch
            self._aspect_ratio_sampler.reset()
            result = self._aspect_ratio_sampler.get_batch(self.batch_size)
            reset_attempts += 1

        if result is None:
            # Sampler is misconfigured or empty - log and stop iteration
            logger.warning(
                f"Aspect ratio sampler exhausted {max_reset_attempts} reset attempts "
                f"without producing batches. Check sampler configuration and dataset. Stopping iteration."
            )
            raise StopIteration()

        example_indices, target_resolution = result
        self._current_target_resolution = target_resolution

        # Update position tracking
        self._position += len(example_indices)
        self.num_iterations += 1

        # Get examples using indices
        examples = [self.dataset.examples[i] for i in example_indices]

        return Batch(examples=examples, rng=self.rng)

    def to_dict(self) -> dict[str, Any]:
        # Serialize RNG state with labeled keys for clarity and robustness
        rng_state = self.rng.getstate()
        return {
            "position": self._position,
            "epoch": self._epoch,
            "num_iterations": self.num_iterations,
            "current_permutation": self._current_permutation,
            "rng_state": {
                "version": rng_state[0],
                "state": list(rng_state[1]),  # tuple -> list for JSON
                "index": rng_state[2],
            },
            "batch_size": self.batch_size,
            "num_epochs": self.num_epochs,
            "start_date_time": self.start_date_time.strftime("%Y-%m-%d %H:%M:%S"),
            "use_aspect_ratio_bucketing": self.use_aspect_ratio_bucketing,
        }

    @classmethod
    def from_dict(cls, state_dict: dict[str, Any], dataset: Dataset) -> "Iterator":
        # Validate and convert the RNG state elements to tuples
        rng_state_raw = state_dict.get("rng_state")
        rng_state = None

        if rng_state_raw is not None:
            try:
                if isinstance(rng_state_raw, dict):
                    # New format with labeled keys (version, state, index)
                    if not all(k in rng_state_raw for k in ("version", "state", "index")):
                        raise ValueError(
                            f"Invalid RNG state dict format: missing keys. Got {list(rng_state_raw.keys())}"
                        )
                    rng_state = (
                        rng_state_raw["version"],
                        tuple(rng_state_raw["state"]),
                        rng_state_raw["index"],
                    )
                elif isinstance(rng_state_raw, (list, tuple)) and len(rng_state_raw) == 3:
                    # Legacy format: [version, state_tuple, index]
                    if not isinstance(rng_state_raw[1], (list, tuple)):
                        raise ValueError(f"Invalid RNG state[1]: expected list/tuple, got {type(rng_state_raw[1])}")
                    rng_state = (rng_state_raw[0], tuple(rng_state_raw[1]), rng_state_raw[2])
                else:
                    raise ValueError(
                        f"Invalid RNG state format: expected dict or list/tuple of 3 elements, got {type(rng_state_raw)}"
                    )
            except (TypeError, IndexError, KeyError) as e:
                raise ValueError(f"Failed to reconstruct RNG state: {e}") from e

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
            use_aspect_ratio_bucketing=state_dict.get("use_aspect_ratio_bucketing", False),
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
        """Get a batch for validation loss computation.

        Uses deterministic random sampling based on epoch to avoid bias from
        always using the first N examples (which could be sorted by quality/difficulty).
        Sampling is reproducible within an epoch for consistent validation metrics.
        """
        examples = self.dataset.examples

        if len(examples) > VALIDATION_BATCH_MAX_SIZE:
            # Use deterministic sampling based on epoch to avoid bias
            # but keep consistent within an epoch for reproducibility
            validation_rng = random.Random(self._epoch)
            examples = validation_rng.sample(examples, VALIDATION_BATCH_MAX_SIZE)

        return Batch(examples=examples, rng=self.rng)

    def total_number_of_steps(self) -> int | None:
        """Calculate total number of training steps.

        Returns:
            Total steps (examples * epochs), or None if num_epochs is None (infinite training).
            Returning None allows tqdm to display an indeterminate progress bar.
        """
        if self.num_epochs is None:
            # For infinite training, return None to let tqdm handle indeterminate progress
            return None
        return self.total_examples * self.num_epochs

    def save(self, path: Path) -> None:
        self.to_json(str(path))
