import os

import numpy as np
from PIL import Image

from mflux import Config, Flux1Controlnet, ModelConfig
from tests.image_generation.helpers.image_generation_test_helper import ImageGeneratorTestHelper


class ImageGeneratorControlnetTestHelper:
    @staticmethod
    def assert_matches_reference_image(
        reference_image_path: str,
        output_image_path: str,
        controlnet_image_path: str,
        model_config: ModelConfig,
        prompt: str,
        steps: int,
        seed: int,
        controlnet_strength: float,
        lora_paths: list[str] | None = None,
        lora_scales: list[float] | None = None,
    ):
        # resolve paths
        reference_image_path = ImageGeneratorTestHelper.resolve_path(reference_image_path)
        output_image_path = ImageGeneratorTestHelper.resolve_path(output_image_path)
        controlnet_image_path = str(ImageGeneratorTestHelper.resolve_path(controlnet_image_path))
        lora_paths = [str(ImageGeneratorTestHelper.resolve_path(p)) for p in lora_paths] if lora_paths else None

        try:
            # given
            flux = Flux1Controlnet(
                model_config=model_config,
                quantize=8,
                lora_paths=lora_paths,
                lora_scales=lora_scales,
            )

            # when
            image = flux.generate_image(
                seed=seed,
                prompt=prompt,
                controlnet_image_path=controlnet_image_path,
                config=Config(
                    num_inference_steps=steps,
                    height=768,
                    width=493,
                    controlnet_strength=controlnet_strength,
                ),
            )
            image.save(path=output_image_path)

            # then
            mse = ImageGeneratorTestHelper.mse(
                Image.open(output_image_path),
                Image.open(reference_image_path),
            )

            assert mse < 5.0, f"Generated image doesn't match reference image (MSE: {mse})"

        finally:
            # cleanup
            if os.path.exists(output_image_path):
                os.remove(output_image_path)

    @staticmethod
    def mse(image1: Image, image2: Image) -> float:
        return np.mean(np.square(np.array(image1) - np.array(image2)))
