import os

from mflux.models.common.config import ModelConfig
from mflux.models.flux.variants.controlnet.flux_controlnet import Flux1Controlnet
from mflux.utils.image_compare import ImageCompare
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
        height: int,
        width: int,
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
                num_inference_steps=steps,
                height=height,
                width=width,
                controlnet_strength=controlnet_strength,
            )
            image.save(path=output_image_path, overwrite=True)

            # then
            ImageCompare.check_images_close_enough(
                output_image_path,
                reference_image_path,
                "Generated controlnet image doesn't match reference image",
            )

        finally:
            # cleanup
            if os.path.exists(output_image_path) and "MFLUX_PRESERVE_TEST_OUTPUT" not in os.environ:
                os.remove(output_image_path)
