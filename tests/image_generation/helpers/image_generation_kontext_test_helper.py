import os
from pathlib import Path

import numpy as np
from PIL import Image

from mflux.config.config import Config
from mflux.config.model_config import ModelConfig
from mflux.kontext.flux_kontext import Flux1Kontext


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
            image.save(path=output_image_path)

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
