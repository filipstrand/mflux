import os
from pathlib import Path

from mflux.models.common.config import ModelConfig
from mflux.models.flux.variants.in_context.flux_in_context_fill import Flux1InContextFill
from mflux.models.flux.variants.in_context.utils.in_context_loras import IC_EDIT_LORA_SCALE, get_ic_edit_lora_path
from mflux.utils.image_compare import ImageCompare


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

        # Build lora_paths: IC-Edit LoRA (required) + user-provided LoRAs
        all_lora_paths = [get_ic_edit_lora_path()]
        all_lora_scales = [IC_EDIT_LORA_SCALE]
        if lora_paths:
            all_lora_paths.extend(lora_paths)
            all_lora_scales.extend(lora_scales or [1.0] * len(lora_paths))

        try:
            # given
            flux = Flux1InContextFill(
                model_config=ModelConfig.dev_fill(),
                quantize=8,
                lora_paths=all_lora_paths,
                lora_scales=all_lora_scales,
            )

            # when
            image = flux.generate_image(
                seed=seed,
                prompt=prompt,
                left_image_path=str(reference_image),
                num_inference_steps=steps,
                height=height or 1024,
                width=width or 1024,
                right_image_path=None,
            )

            # Save the result
            image.get_right_half().save(path=output_image_path, overwrite=True)

            # then
            ImageCompare.check_images_close_enough(
                output_image_path,
                reference_image_path,
                "Generated ic-edit image doesn't match reference image.",
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
