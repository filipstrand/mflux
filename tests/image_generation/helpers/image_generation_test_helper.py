import os
from pathlib import Path

import numpy as np
from PIL import Image

from mflux.config.config import Config
from mflux.config.model_config import ModelConfig
from mflux.flux.flux import Flux1


class ImageGeneratorTestHelper:
    @staticmethod
    def assert_matches_reference_image(
        reference_image_path: str,
        output_image_path: str,
        model_config: ModelConfig,
        prompt: str,
        steps: int,
        seed: int,
        height: int | None = None,
        width: int | None = None,
        image_path: str | None = None,
        image_strength: float | None = None,
        lora_paths: list[str] | None = None,
        lora_scales: list[float] | None = None,
    ):
        # resolve paths
        reference_image_path = ImageGeneratorTestHelper.resolve_path(reference_image_path)
        output_image_path = ImageGeneratorTestHelper.resolve_path(output_image_path)
        lora_paths = [str(ImageGeneratorTestHelper.resolve_path(p)) for p in lora_paths] if lora_paths else None

        try:
            # given
            flux = Flux1(
                model_config=model_config,
                quantize=8,
                lora_paths=lora_paths,
                lora_scales=lora_scales,
            )
            # when
            image = flux.generate_image(
                seed=seed,
                prompt=prompt,
                config=Config(
                    num_inference_steps=steps,
                    image_path=ImageGeneratorTestHelper.resolve_path(image_path),
                    image_strength=image_strength,
                    height=height,
                    width=width,
                ),
            )
            image.save(path=output_image_path, overwrite=True)

            # then
            np.testing.assert_array_equal(
                np.array(Image.open(output_image_path)),
                np.array(Image.open(reference_image_path)),
                err_msg=f"Generated image doesn't match reference image. Check {output_image_path} vs {reference_image_path}",
            )

        finally:
            # cleanup
            if os.path.exists(output_image_path):
                os.remove(output_image_path)

    @staticmethod
    def resolve_path(path) -> Path | None:
        if path is None:
            return None
        return Path(__file__).parent.parent.parent / "resources" / path
