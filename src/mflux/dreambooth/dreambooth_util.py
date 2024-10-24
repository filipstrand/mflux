from pathlib import Path

import mlx.core as mx
from mlx.utils import tree_flatten

from mflux import Flux1


class DreamBoothUtil:
    @staticmethod
    def track_progress(loss: mx.float16, t: int) -> None:
        print(f"Loss: {loss}")

    @staticmethod
    def save_incrementally(flux: Flux1, t: int) -> None:
        if t % 50:
            DreamBoothUtil.save_adapter(flux, t)

    @staticmethod
    def save_adapter(flux: Flux1, t: int) -> None:
        iteration = t
        out_dir = Path("/Users/filipstrand/Desktop/test")
        out_dir.mkdir(parents=True, exist_ok=True)
        out_file = out_dir / f"{iteration:07d}_adapters.safetensors"
        print(f"Saving {str(out_file)}")

        mx.save_safetensors(
            str(out_file),
            dict(tree_flatten(flux.trainable_parameters())),
            metadata={
                "lora_rank": "4",
                "lora_blocks": "1",
            },
        )
