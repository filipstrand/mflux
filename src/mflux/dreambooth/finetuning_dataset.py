import json
from pathlib import Path

import mlx.core as mx
import PIL.Image
from mlx import nn

from mflux import Flux1, ImageUtil
from mflux.dreambooth.finetuning_batch_example import Example
from mflux.dreambooth.finetuning_dataset_iterator import DatasetIterator
from mflux.post_processing.array_util import ArrayUtil


class FineTuningDataset:
    def __init__(self, raw_data: list[dict[str, str]], root_dir: Path):
        self.raw_data = raw_data
        self.root_dir = root_dir
        self.prepared_dataset = None

    @staticmethod
    def load_from_disk(path: str | Path) -> "FineTuningDataset":
        path = Path(path)
        index_file = path / "index.json"

        with open(index_file, "r") as f:
            data = json.load(f)

        return FineTuningDataset(raw_data=data, root_dir=path)

    def prepare_dataset(self, flux: Flux1) -> None:
        examples = self.create_examples(flux, self.raw_data)
        self.prepared_dataset = examples

    def get_iterator(self, batch_size: int = 1) -> iter:
        return iter(DatasetIterator(self, batch_size=batch_size))

    def create_examples(self, flux: Flux1, raw_data: list[dict[str, str]]) -> list[Example]:
        examples = []
        for entry in raw_data:
            # Encode the image
            image_path = self.root_dir / entry["image"]
            image = PIL.Image.open(image_path)
            encoded_image = FineTuningDataset._encode_image(flux.vae, image)

            # Encode the prompt
            prompt = entry["prompt"]
            prompt_embeds = flux.t5_text_encoder.forward(flux.t5_tokenizer.tokenize(prompt))
            pooled_prompt_embeds = flux.clip_text_encoder.forward(flux.clip_tokenizer.tokenize(prompt))
            example = Example(
                prompt=prompt,
                image=image,
                encoded_image=encoded_image,
                prompt_embeds=prompt_embeds,
                pooled_prompt_embeds=pooled_prompt_embeds,
            )
            examples.append(example)
        return examples

    @staticmethod
    def _encode_image(vae: nn.Module, image: PIL.Image.Image) -> mx.array:
        scaled_user_image = ImageUtil.scale_to_dimensions(image, target_width=1024, target_height=1024)
        encoded = vae.encode(ImageUtil.to_array(scaled_user_image))
        latents = ArrayUtil.pack_latents(encoded, width=1024, height=1024)
        return latents
