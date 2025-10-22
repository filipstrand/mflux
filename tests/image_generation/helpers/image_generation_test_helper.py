import os
from pathlib import Path
from typing import Type, Union

from mflux.config.config import Config
from mflux.config.model_config import ModelConfig
from mflux.models.qwen.variants.txt2img.qwen_image import QwenImage

from .image_compare import check_images_close_enough


# Lazy import Flux1 to avoid import errors when only running Qwen tests
def _get_flux1():
    from mflux.models.flux.variants.txt2img.flux import Flux1

    return Flux1


class ImageGeneratorTestHelper:
    @staticmethod
    def assert_matches_reference_image(
        reference_image_path: str,
        output_image_path: str,
        model_class: Type[Union[QwenImage]],
        model_config: ModelConfig,
        prompt: str,
        steps: int,
        seed: int,
        quantize=8,
        height: int | None = None,
        width: int | None = None,
        image_path: str | None = None,
        image_strength: float | None = None,
        lora_paths: list[str] | None = None,
        lora_scales: list[float] | None = None,
        lora_names: list[str] | None = None,
        lora_repo_id: str | None = None,
        negative_prompt: str | None = None,
        guidance: float | None = None,
    ):
        # resolve paths
        reference_image_path = ImageGeneratorTestHelper.resolve_path(reference_image_path)
        output_image_path = ImageGeneratorTestHelper.resolve_path(output_image_path)
        lora_paths = [str(ImageGeneratorTestHelper.resolve_path(p)) for p in lora_paths] if lora_paths else None

        try:
            # given
            model_kwargs = {
                "model_config": model_config,
                "quantize": quantize,
                "lora_paths": lora_paths,
                "lora_scales": lora_scales,
            }

            # Add HuggingFace LoRA parameters if provided
            if lora_names is not None:
                model_kwargs["lora_names"] = lora_names
            if lora_repo_id is not None:
                model_kwargs["lora_repo_id"] = lora_repo_id

            model = model_class(**model_kwargs)
            config_kwargs = {
                "num_inference_steps": steps,
                "image_path": ImageGeneratorTestHelper.resolve_path(image_path),
                "image_strength": image_strength,
                "height": height,
                "width": width,
            }

            # Add guidance if provided
            if guidance is not None:
                config_kwargs["guidance"] = guidance

            generate_kwargs = {
                "seed": seed,
                "prompt": prompt,
                "config": Config(**config_kwargs),
            }

            # Add negative_prompt for Qwen models
            if model_class == QwenImage and negative_prompt is not None:
                generate_kwargs["negative_prompt"] = negative_prompt

            # when
            image = model.generate_image(**generate_kwargs)
            image.save(path=output_image_path, overwrite=True)

            # then
            check_images_close_enough(
                output_image_path,
                reference_image_path,
                "Generated image doesn't match reference image.",
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
