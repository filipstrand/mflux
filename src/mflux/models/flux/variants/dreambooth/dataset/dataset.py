from pathlib import Path

import mlx.core as mx
import PIL.Image
from mlx import nn
from tqdm import tqdm

from mflux.models.flux.latent_creator.flux_latent_creator import FluxLatentCreator
from mflux.models.flux.variants.dreambooth.dataset.batch import Example
from mflux.models.flux.variants.dreambooth.dataset.dreambooth_preprocessing import DreamBoothPreProcessing
from mflux.models.flux.variants.dreambooth.state.training_spec import ExampleSpec
from mflux.models.flux.variants.txt2img.flux import Flux1
from mflux.utils.image_util import ImageUtil


class Dataset:
    def __init__(self, examples: list[Example]):
        self.examples = examples

    @staticmethod
    def prepare_dataset(
        flux: Flux1,
        raw_data: list[ExampleSpec],
        width: int,
        height: int,
    ) -> "Dataset":
        # Encode the original examples (image and text)
        examples = Dataset._create_examples(flux, raw_data, width=width, height=height)

        # Expend the original dataset to get more training data with variations
        augmented_examples = [
            variation for example in examples for variation in DreamBoothPreProcessing.augment(example)
        ]
        # Dataset is now prepared
        return Dataset(augmented_examples)

    def size(self) -> int:
        return len(self.examples)

    @staticmethod
    def _create_examples(
        flux: Flux1,
        raw_data: list[ExampleSpec],
        width: int,
        height: int,
    ) -> list[Example]:
        examples = []
        for i, entry in enumerate(tqdm(raw_data, desc="Encoding original dataset")):
            # Encode the image
            encoded_image = Dataset._encode_image(flux.vae, entry.image, width=width, height=height)

            # Encode the prompt
            t5_output = flux.tokenizers["t5"].tokenize(entry.prompt)
            clip_output = flux.tokenizers["clip"].tokenize(entry.prompt)
            prompt_embeds = flux.t5_text_encoder(t5_output.input_ids)
            pooled_prompt_embeds = flux.clip_text_encoder(clip_output.input_ids)

            # Create the example object
            example = Example(
                example_id=i,
                prompt=entry.prompt,
                image_path=entry.image,
                encoded_image=encoded_image,
                prompt_embeds=prompt_embeds,
                pooled_prompt_embeds=pooled_prompt_embeds,
            )
            examples.append(example)

            # Evaluate to enable progress tracking
            mx.eval(encoded_image, prompt_embeds, pooled_prompt_embeds)

        return examples

    @staticmethod
    def _encode_image(vae: nn.Module, image_path: Path, width: int, height: int) -> mx.array:
        image = PIL.Image.open(image_path.resolve()).convert("RGB")
        scaled_user_image = ImageUtil.scale_to_dimensions(image, target_width=width, target_height=height)
        encoded = vae.encode(ImageUtil.to_array(scaled_user_image))
        latents = FluxLatentCreator.pack_latents(encoded, width=width, height=height)
        return latents
