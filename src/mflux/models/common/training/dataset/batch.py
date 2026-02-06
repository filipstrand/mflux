from __future__ import annotations

from pathlib import Path
from random import Random
from typing import Any

import mlx.core as mx


class DataItem:
    def __init__(
        self,
        data_id: int,
        prompt: str,
        image_path: str | Path,
        clean_latents: mx.array,
        cond: Any,
        width: int,
        height: int,
    ):
        self.data_id = data_id
        self.prompt = prompt
        self.image_name = str(image_path)
        self.clean_latents = clean_latents
        self.cond = cond
        self.width = int(width)
        self.height = int(height)


class Batch:
    def __init__(self, data: list[DataItem], rng: Random):
        self.rng = rng
        self.data = data
