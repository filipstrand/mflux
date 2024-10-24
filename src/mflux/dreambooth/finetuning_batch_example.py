import mlx.core as mx
import PIL.Image


class Example:
    def __init__(
        self,
        prompt: str,
        image: PIL.Image.Image,
        encoded_image: mx.array,
        prompt_embeds: mx.array,
        pooled_prompt_embeds: mx.array,
    ):
        self.prompt = prompt
        self.image = image
        self.encoded_image_latents = encoded_image
        self.prompt_embeds = prompt_embeds
        self.pooled_prompt_embeds = pooled_prompt_embeds


class Batch:
    def __init__(self, examples: list[Example]):
        self.examples = examples
