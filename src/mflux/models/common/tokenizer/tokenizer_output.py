from dataclasses import dataclass

import mlx.core as mx


@dataclass
class TokenizerOutput:
    input_ids: mx.array
    attention_mask: mx.array
    pixel_values: mx.array | None = None
    image_grid_thw: mx.array | None = None
