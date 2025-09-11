import os
from pathlib import Path

from mflux.config.config import Config
from mflux.config.model_config import ModelConfig
from mflux.models.flux.variants.redux.flux_redux import Flux1Redux

from .image_compare import check_images_close_enough


class ImageGeneratorReduxTestHelper:
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
        redux_image_path: str,
    ):
        # resolve paths
        reference_image_path = ImageGeneratorReduxTestHelper.resolve_path(reference_image_path)
        output_image_path = ImageGeneratorReduxTestHelper.resolve_path(output_image_path)
        redux_image_path = ImageGeneratorReduxTestHelper.resolve_path(redux_image_path)

        try:
            # given
            flux = Flux1Redux(
                model_config=model_config,
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
                    redux_image_paths=[redux_image_path],
                ),
            )
            image.save(path=output_image_path, overwrite=True)

            # then
            check_images_close_enough(
                output_image_path,
                reference_image_path,
                "Generated redux image doesn't match reference image.",
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
