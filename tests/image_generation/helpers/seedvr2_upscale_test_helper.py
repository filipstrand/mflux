import os

from mflux.models.common.config.model_config import ModelConfig
from mflux.models.common.vae.tiling_config import TilingConfig
from mflux.models.seedvr2.variants.upscale.seedvr2 import SeedVR2
from mflux.utils.image_compare import ImageCompare
from tests.image_generation.helpers.image_generation_test_helper import ImageGeneratorTestHelper


class SeedVR2UpscaleTestHelper:
    @staticmethod
    def assert_matches_reference_image(
        reference_image_path: str,
        output_image_path: str,
        input_image_path: str,
        *,
        tiling_config: TilingConfig | None = None,
        seed: int = 42,
        resolution: int = 320,
        quantize: int | None = None,
        mismatch_threshold: float | None = None,
    ) -> None:
        reference_image_path = ImageGeneratorTestHelper.resolve_path(reference_image_path)
        output_image_path = ImageGeneratorTestHelper.resolve_path(output_image_path)
        input_image_path = str(ImageGeneratorTestHelper.resolve_path(input_image_path))

        try:
            model = SeedVR2(
                quantize=quantize,
                model_path=None,
                model_config=ModelConfig.seedvr2_3b(),
            )
            if tiling_config is not None:
                model.tiling_config = tiling_config

            result = model.generate_image(
                image_path=input_image_path,
                seed=seed,
                resolution=resolution,
            )
            result.save(output_image_path)

            ImageCompare.check_images_close_enough(
                output_image_path,
                reference_image_path,
                "Generated SeedVR2 upscaled image doesn't match reference image.",
                mismatch_threshold=mismatch_threshold,
            )
        finally:
            if os.path.exists(output_image_path) and "MFLUX_PRESERVE_TEST_OUTPUT" not in os.environ:
                os.remove(output_image_path)
