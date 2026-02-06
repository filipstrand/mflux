from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import overload

from mflux.models.common.training.dataset.batch import DataItem
from mflux.models.common.training.dataset.data_cache import CachePaths, TrainingDataCache
from mflux.models.common.training.state.training_spec import DataSpec


class DiskBackedData(Sequence[DataItem]):
    def __init__(self, *, data_specs: list[DataSpec], cache_paths: CachePaths):
        self._data_specs = data_specs
        self._cache_paths = cache_paths

    def __len__(self) -> int:  # noqa: D401
        return len(self._data_specs)

    @overload
    def __getitem__(self, idx: int) -> DataItem: ...

    @overload
    def __getitem__(self, idx: slice) -> Sequence[DataItem]: ...

    def __getitem__(self, idx: int | slice) -> DataItem | Sequence[DataItem]:
        if isinstance(idx, slice):
            start, stop, step = idx.indices(len(self))
            return [self[i] for i in range(start, stop, step)]

        if idx < 0:
            idx = len(self._data_specs) + idx
        if idx < 0 or idx >= len(self._data_specs):
            raise IndexError(idx)

        spec = self._data_specs[idx]
        clean_latents, cond, width, height = TrainingDataCache.load_tensors(paths=self._cache_paths, data_id=idx)
        return DataItem(
            data_id=idx,
            prompt=spec.prompt,
            image_path=Path(spec.image),
            clean_latents=clean_latents,
            cond=cond,
            width=width,
            height=height,
        )
