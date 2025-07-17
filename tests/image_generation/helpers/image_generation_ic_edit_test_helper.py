import os
from pathlib import Path

import numpy as np
from PIL import Image

from mflux.community.in_context.flux_in_context_fill import Flux1InContextFill
from mflux.community.in_context.utils.in_context_loras import prepare_ic_edit_loras
from mflux.config.config import Config
from mflux.config.model_config import ModelConfig


class ImageGeneratorICEditTestHelper:
    @staticmethod
    def assert_matches_reference_image(
        reference_image_path: str,
        output_image_path: str,
        prompt: str,
        steps: int,
        seed: int,
        height: int | None = None,
        width: int | None = None,
        reference_image: str | None = None,
        lora_paths: list[str] | None = None,
        lora_scales: list[float] | None = None,
    ):
        # resolve paths
        reference_image_path = ImageGeneratorICEditTestHelper.resolve_path(reference_image_path)
        output_image_path = ImageGeneratorICEditTestHelper.resolve_path(output_image_path)
        reference_image = ImageGeneratorICEditTestHelper.resolve_path(reference_image)
        lora_paths = [str(ImageGeneratorICEditTestHelper.resolve_path(p)) for p in lora_paths] if lora_paths else None

        try:
            # given
            flux = Flux1InContextFill(
                model_config=ModelConfig.dev_fill(),
                quantize=8,
                lora_paths=prepare_ic_edit_loras(lora_paths),
                lora_scales=lora_scales,
            )

            # when
            image = flux.generate_image(
                seed=seed,
                prompt=prompt,
                left_image_path=str(reference_image),
                right_image_path=None,
                config=Config(
                    num_inference_steps=steps,
                    height=height,
                    width=width,
                ),
            )

            # Save the result
            image.get_right_half().save(path=output_image_path, overwrite=True)

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
