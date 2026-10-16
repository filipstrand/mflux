from pathlib import Path

import mlx.core as mx
import PIL.Image

from mflux.models.flux3.latent_creator.flux3_latent_creator import Flux3LatentCreator
from mflux.models.flux3.latent_creator.flux3_reference_images import Flux3ReferenceImages
from mflux.models.flux3.variants.flux3_klein_core import _Flux3KleinCore
from mflux.utils.generated_image import GeneratedImage


class Flux3KleinEdit(_Flux3KleinCore):
    def generate_image(
        self,
        seed: int,
        prompt: str,
        image_paths: list[Path | str],
        num_inference_steps: int | None = None,
        height: int | None = None,
        width: int | None = None,
        guidance: float | None = None,
    ) -> GeneratedImage:
        # width/height None -> the first prepared reference's size; a single reference prepared with neither
        # given (or with exactly its fitted size) is cropped back to its fitted box (as the reference does).
        images = [PIL.Image.open(path) for path in image_paths]
        references, box = Flux3ReferenceImages.prepare(images)
        width, height, crop_box = Flux3ReferenceImages.resolve_canvas(references, box, width, height)
        cond_tokens, cond_ids = self.encode_references(references)
        return self._generate(
            seed=seed,
            prompt=prompt,
            num_inference_steps=num_inference_steps,
            height=height,
            width=width,
            guidance=guidance,
            cond_tokens=cond_tokens,
            cond_ids=cond_ids,
            image_paths=image_paths,
            crop_box=crop_box,
        )

    def encode_references(self, references: list[PIL.Image.Image]) -> tuple[mx.array, mx.array]:
        tokens, ids = [], []
        for index, reference in enumerate(references):
            image = mx.array(Flux3ReferenceImages.to_array(reference))
            encoded = Flux3LatentCreator.encode_tokens(self.vae, image, tiling_config=self.tiling_config)
            h, w = Flux3LatentCreator.latent_shape(reference.height, reference.width)
            tokens.append(encoded.astype(mx.bfloat16))
            ids.append(Flux3LatentCreator.prepare_ids(h, w, layer=-(1 + index)))
        cond_tokens = mx.concatenate(tokens, axis=1)
        cond_ids = mx.concatenate(ids, axis=0)
        mx.eval(cond_tokens, cond_ids)
        return cond_tokens, cond_ids
