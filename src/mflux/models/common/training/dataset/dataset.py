from __future__ import annotations

from collections.abc import Sequence

from mflux.models.common.training.dataset.batch import DataItem


class Dataset:
    def __init__(self, data: Sequence[DataItem]):
        self.data = data

    def size(self) -> int:
        return len(self.data)
