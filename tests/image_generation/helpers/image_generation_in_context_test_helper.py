import os
from pathlib import Path

from mflux.models.common.config import ModelConfig
from mflux.models.flux.variants.in_context.flux_in_context_dev import Flux1InContextDev
from mflux.models.flux.variants.in_context.utils.in_context_loras import get_lora_path
from mflux.utils.image_compare import ImageCompare


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

        # Build lora_paths: style LoRA (if specified) + user-provided LoRAs
        all_lora_paths = []
        all_lora_scales = []
        if lora_style:
            all_lora_paths.append(get_lora_path(lora_style))
            all_lora_scales.append(1.0)
        if lora_paths:
            all_lora_paths.extend(lora_paths)
            all_lora_scales.extend(lora_scales or [1.0] * len(lora_paths))

        try:
            # given
            flux = Flux1InContextDev(
                model_config=model_config,
                quantize=8,
                lora_paths=all_lora_paths or None,
                lora_scales=all_lora_scales or None,
            )
            # when
            image = flux.generate_image(
                seed=seed,
                prompt=prompt,
                num_inference_steps=steps,
                image_path=image_path,
                height=height or 1024,
                width=width or 1024,
            )
            # Save only the right half of the image (the generated part)
            image.get_right_half().save(path=output_image_path, overwrite=True)

            # then
            ImageCompare.check_images_close_enough(
                output_image_path,
                reference_image_path,
                "Generated in-context image doesn't match reference image.",
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
