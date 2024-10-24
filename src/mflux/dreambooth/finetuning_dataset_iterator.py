import random
from typing import TYPE_CHECKING

from mflux.dreambooth.finetuning_batch_example import Batch

if TYPE_CHECKING:
    from mflux.dreambooth.finetuning_dataset import FineTuningDataset


class DatasetIterator:
    def __init__(self, dataset: "FineTuningDataset", batch_size: int = 1):
        self.dataset = dataset.prepared_dataset
        self.batch_size = batch_size
        self.indices = list(range(len(self.dataset)))
        random.shuffle(self.indices)
        self.current_idx = 0

    def __iter__(self):
        return self

    def __next__(self) -> Batch:
        if self.current_idx >= len(self.indices):
            # Reshuffle indices when we've gone through all examples
            random.shuffle(self.indices)
            self.current_idx = 0

        # Get batch_size number of random examples
        batch_indices = self.indices[self.current_idx : self.current_idx + self.batch_size]
        self.current_idx += self.batch_size

        batch_examples = [self.dataset[i] for i in batch_indices]
        return Batch(examples=batch_examples)
