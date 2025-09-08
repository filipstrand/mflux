import os

from mflux.config.config import Config
from mflux.config.model_config import ModelConfig
from mflux.flux_tools.fill.flux_fill import Flux1Fill
from mflux.ui import defaults as ui_defaults
from tests.image_generation.helpers.image_generation_test_helper import ImageGeneratorTestHelper

from .image_compare import check_images_close_enough


class ImageGeneratorFillTestHelper:
    @staticmethod
    def assert_matches_reference_image(
        reference_image_path: str,
        output_image_path: str,
        model_config: ModelConfig,
        steps: int,
        seed: int,
        height: int,
        width: int,
        prompt: str,
        image_path: str,
        masked_image_path: str,
        lora_paths: list[str] | None = None,
        lora_scales: list[float] | None = None,
    ):
        # resolve paths
        reference_image_path = ImageGeneratorTestHelper.resolve_path(reference_image_path)
        output_image_path = ImageGeneratorTestHelper.resolve_path(output_image_path)
        image_path = str(ImageGeneratorTestHelper.resolve_path(image_path))
        masked_image_path = str(ImageGeneratorTestHelper.resolve_path(masked_image_path))
        lora_paths = [str(ImageGeneratorTestHelper.resolve_path(p)) for p in lora_paths] if lora_paths else None

        try:
            # given
            flux = Flux1Fill(
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
                    height=height,
                    width=width,
                    image_path=image_path,
                    masked_image_path=masked_image_path,
                    guidance=ui_defaults.DEFAULT_DEV_FILL_GUIDANCE,
                ),
            )
            image.save(path=output_image_path, overwrite=True)

            # then
            check_images_close_enough(
                output_image_path,
                reference_image_path,
                "Generated fill image doesn't match reference image.",
            )

        finally:
            # cleanup
            if os.path.exists(output_image_path) and "MFLUX_PRESERVE_TEST_OUTPUT" not in os.environ:
                os.remove(output_image_path)
