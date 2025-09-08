import os
from pathlib import Path

from mflux.config.config import Config
from mflux.config.model_config import ModelConfig
from mflux.models.flux.variants.depth.flux_depth import Flux1Depth

from .image_compare import check_images_close_enough


class ImageGeneratorDepthTestHelper:
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
        depth_image_path: str | None = None,
    ):
        # resolve paths
        reference_image_path = ImageGeneratorDepthTestHelper.resolve_path(reference_image_path)
        output_image_path = ImageGeneratorDepthTestHelper.resolve_path(output_image_path)
        depth_image_path = ImageGeneratorDepthTestHelper.resolve_path(depth_image_path)
        image_path = ImageGeneratorDepthTestHelper.resolve_path(image_path)

        try:
            # given
            flux = Flux1Depth(quantize=8)
            image = flux.generate_image(
                seed=seed,
                prompt=prompt,
                config=Config(
                    num_inference_steps=steps,
                    height=height,
                    width=width,
                    image_path=image_path,
                    depth_image_path=depth_image_path,
                ),
            )

            image.save(path=output_image_path, overwrite=True)

            # then
            check_images_close_enough(
                output_image_path,
                reference_image_path,
                "Generated depth image doesn't match reference image.",
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
