import os
from pathlib import Path

from mflux.config.config import Config
from mflux.config.model_config import ModelConfig
from mflux.models.flux.variants.kontext.flux_kontext import Flux1Kontext

from .image_compare import check_images_close_enough


class ImageGeneratorKontextTestHelper:
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
        kontext_image_path: str,
        guidance: float = 2.5,
    ):
        # resolve paths
        reference_image_path = ImageGeneratorKontextTestHelper.resolve_path(reference_image_path)
        output_image_path = ImageGeneratorKontextTestHelper.resolve_path(output_image_path)
        kontext_image_path = ImageGeneratorKontextTestHelper.resolve_path(kontext_image_path)

        try:
            # given
            flux = Flux1Kontext(
                quantize=8,
            )

            # when
            image = flux.generate_image(
                seed=seed,
                prompt=prompt,
                config=Config(
                    num_inference_steps=steps,
                    height=height,
                    width=width,
                    guidance=guidance,
                    image_path=kontext_image_path,
                ),
            )
            image.save(path=output_image_path, overwrite=True)

            # then
            check_images_close_enough(
                output_image_path,
                reference_image_path,
                "Generated kontext image doesn't match reference image.",
            )
        finally:
            # cleanup
            if os.path.exists(output_image_path) and "MFLUX_PRESERVE_TEST_OUTPUT" not in os.environ:
                os.remove(output_image_path)

    @staticmethod
    def resolve_path(path) -> Path | None:
        if path is None:
            return None
        return Path(__file__).parent.parent.parent / "resources" / path
