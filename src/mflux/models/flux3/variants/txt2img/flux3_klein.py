from mflux.models.flux3.variants.flux3_klein_core import _Flux3KleinCore
from mflux.utils.generated_image import GeneratedImage


class Flux3Klein(_Flux3KleinCore):
    def generate_image(
        self,
        seed: int,
        prompt: str,
        num_inference_steps: int | None = None,
        height: int = 1024,
        width: int = 1024,
        guidance: float | None = None,
    ) -> GeneratedImage:
        return self._generate(
            seed=seed,
            prompt=prompt,
            num_inference_steps=num_inference_steps,
            height=height,
            width=width,
            guidance=guidance,
        )
