from pathlib import Path
from random import Random

import mlx.core as mx


class Example:
    def __init__(
        self,
        example_id: int,
        prompt: str,
        image_path: str | Path,
        encoded_image: mx.array,
        prompt_embeds: mx.array,
        pooled_prompt_embeds: mx.array,
    ):
        self.example_id = example_id
        self.prompt = prompt
        self.image_name = str(image_path)
        self.clean_latents = encoded_image
        self.prompt_embeds = prompt_embeds
        self.pooled_prompt_embeds = pooled_prompt_embeds


class Batch:
    def __init__(self, examples: list[Example], rng: Random):
        self.rng = rng
        self.examples = examples
