import os
from pathlib import Path
from typing import Optional

from mflux.models.fibo.variants.txt2img.fibo import FIBO
from mflux.utils.image_compare import ImageCompare


class ImageGeneratorFiboTestHelper:
    @staticmethod
    def assert_matches_reference_image(
        reference_image_path: str,
        output_image_path: str,
        prompt: str,
        steps: int,
        seed: int,
        height: int,
        width: int,
        guidance: float = 4.0,
        negative_prompt: Optional[str] = None,
        mismatch_threshold: Optional[float] = None,
        quantize: Optional[int] = None,
    ):
        # resolve paths
        reference_image_path = ImageGeneratorFiboTestHelper.resolve_path(reference_image_path)
        output_image_path = ImageGeneratorFiboTestHelper.resolve_path(output_image_path)

        try:
            # Step 1: Create FIBO model
            model = FIBO(
                quantize=quantize,
                model_path=None,
            )

            # Step 2: Generate image from prompt
            image = model.generate_image(
                seed=seed,
                prompt=prompt,
                num_inference_steps=steps,
                height=height,
                width=width,
                guidance=guidance,
                scheduler="flow_match_euler_discrete",
                negative_prompt=negative_prompt,
            )

            # Step 3: Save output image
            image.save(path=output_image_path, overwrite=True)

            # Step 4: Compare with reference
            ImageCompare.check_images_close_enough(
                output_image_path,
                reference_image_path,
                "Generated image doesn't match reference image.",
                mismatch_threshold=mismatch_threshold,
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
