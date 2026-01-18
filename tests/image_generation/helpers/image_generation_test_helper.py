import gc
import os
from pathlib import Path
from typing import Type, Union

from mflux.callbacks.instances.memory_saver import MemorySaver
from mflux.models.common.config import ModelConfig
from mflux.models.flux.variants.txt2img.flux import Flux1
from mflux.models.qwen.variants.txt2img.qwen_image import QwenImage
from mflux.utils.image_compare import ImageCompare


class ImageGeneratorTestHelper:
    @staticmethod
    def assert_matches_reference_image(
        reference_image_path: str,
        output_image_path: str,
        model_class: Type[Union[Flux1, QwenImage]],
        model_config: ModelConfig,
        prompt: str,
        steps: int,
        seed: int,
        quantize=8,
        height: int | None = None,
        width: int | None = None,
        image_path: str | None = None,
        image_paths: list[str] | None = None,
        image_strength: float | None = None,
        lora_paths: list[str] | None = None,
        lora_scales: list[float] | None = None,
        negative_prompt: str | None = None,
        guidance: float | None = None,
        mismatch_threshold: float | None = None,
        low_memory: bool = False,
    ):
        # resolve paths
        reference_image_path = ImageGeneratorTestHelper.resolve_path(reference_image_path)
        output_image_path = ImageGeneratorTestHelper.resolve_path(output_image_path)
        lora_paths = [str(ImageGeneratorTestHelper.resolve_path(p)) for p in lora_paths] if lora_paths else None

        model = None
        image = None
        memory_saver = None
        try:
            # given
            model_kwargs = {
                "model_config": model_config,
                "quantize": quantize,
                "lora_paths": lora_paths,
                "lora_scales": lora_scales,
            }

            model = model_class(**model_kwargs)

            generate_kwargs = {
                "seed": seed,
                "prompt": prompt,
                "num_inference_steps": steps,
                "image_strength": image_strength,
                "height": height or 1024,
                "width": width or 1024,
            }

            if image_path is not None:
                generate_kwargs["image_path"] = ImageGeneratorTestHelper.resolve_path(image_path)

            if image_paths is not None:
                generate_kwargs["image_paths"] = [ImageGeneratorTestHelper.resolve_path(p) for p in image_paths]

            # Add guidance if provided
            if guidance is not None:
                generate_kwargs["guidance"] = guidance

            # Add negative_prompt for Qwen models
            if model_class == QwenImage and negative_prompt is not None:
                generate_kwargs["negative_prompt"] = negative_prompt

            # Register MemorySaver callback if low_memory mode is enabled
            if low_memory:
                memory_saver = MemorySaver(model=model)
                model.callbacks.register(memory_saver)

            # when
            image = model.generate_image(**generate_kwargs)
            image.save(path=output_image_path, overwrite=True)

            # then
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

            # Help avoid long test-suite runs building up native memory.
            image = None
            model = None
            memory_saver = None
            gc.collect()

            try:
                import mlx.core as mx  # noqa: PLC0415

                mx.metal.clear_cache()
            except (ImportError, AttributeError):
                pass

    @staticmethod
    def resolve_path(path) -> Path | None:
        if path is None:
            return None
        return Path(__file__).parent.parent.parent / "resources" / path
