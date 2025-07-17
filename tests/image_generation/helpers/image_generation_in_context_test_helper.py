import os
from pathlib import Path

import numpy as np
from PIL import Image

from mflux.community.in_context.flux_in_context_dev import Flux1InContextDev
from mflux.community.in_context.utils.in_context_loras import LORA_REPO_ID, get_lora_filename
from mflux.config.config import Config
from mflux.config.model_config import ModelConfig


class ImageGeneratorInContextTestHelper:
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
        lora_style: str | None = None,
        lora_paths: list[str] | None = None,
        lora_scales: list[float] | None = None,
    ):
        # resolve paths
        reference_image_path = ImageGeneratorInContextTestHelper.resolve_path(reference_image_path)
        output_image_path = ImageGeneratorInContextTestHelper.resolve_path(output_image_path)
        image_path = ImageGeneratorInContextTestHelper.resolve_path(image_path)
        lora_paths = (
            [str(ImageGeneratorInContextTestHelper.resolve_path(p)) for p in lora_paths] if lora_paths else None
        )

        try:
            # given
            flux = Flux1InContextDev(
                model_config=model_config,
                quantize=8,
                lora_names=[get_lora_filename(lora_style)] if lora_style else None,
                lora_repo_id=LORA_REPO_ID if lora_style else None,
                lora_paths=lora_paths,
                lora_scales=lora_scales,
            )
            # when
            image = flux.generate_image(
                seed=seed,
                prompt=prompt,
                config=Config(
                    num_inference_steps=steps,
                    image_path=image_path,
                    height=height,
                    width=width,
                ),
            )
            # Save only the right half of the image (the generated part)
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
