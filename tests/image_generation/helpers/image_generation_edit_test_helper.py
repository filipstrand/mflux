import os
from pathlib import Path
from typing import Any, Type

from mflux.callbacks.callback_registry import CallbackRegistry
from mflux.callbacks.instances.stepwise_handler import StepwiseHandler
from mflux.config.config import Config
from mflux.config.model_config import ModelConfig
from mflux.utils.image_compare import ImageCompare


class ImageGeneratorEditTestHelper:
    @staticmethod
    def assert_matches_reference_image(
        reference_image_path: str,
        output_image_path: str,
        model_class: Type[Any],
        model_config: ModelConfig,
        steps: int,
        seed: int,
        height: int,
        width: int,
        prompt: str,
        image_path: str | None = None,
        guidance: float = 2.5,
        negative_prompt: str | None = None,
        quantize: int = 8,
        image_paths: list[str] | None = None,
        lora_names: list[str] | None = None,
        lora_repo_id: str | None = None,
        mismatch_threshold: float | None = None,
    ):
        # resolve paths
        reference_image_path = ImageGeneratorEditTestHelper.resolve_path(reference_image_path)
        output_image_path = ImageGeneratorEditTestHelper.resolve_path(output_image_path)

        # For Qwen Edit, if image_paths is provided but image_path is not, use first element of image_paths
        is_qwen_edit = "QwenImageEdit" in model_class.__name__
        if is_qwen_edit and image_paths and not image_path:
            image_path = image_paths[0]

        if image_path:
            image_path = ImageGeneratorEditTestHelper.resolve_path(image_path)
        if image_paths:
            image_paths = [str(ImageGeneratorEditTestHelper.resolve_path(p)) for p in image_paths]

        try:
            # given
            model_kwargs = {
                "quantize": quantize,
            }

            # Add HuggingFace LoRA parameters if provided
            if lora_names is not None:
                model_kwargs["lora_names"] = lora_names
            if lora_repo_id is not None:
                model_kwargs["lora_repo_id"] = lora_repo_id

            model = model_class(**model_kwargs)

            # when
            config_kwargs = {
                "num_inference_steps": steps,
                "height": height,
                "width": width,
                "guidance": guidance,
                "image_path": image_path,
                "scheduler": "flow_match_euler_discrete",  # Match debug script
            }
            generate_kwargs = {
                "seed": seed,
                "prompt": prompt,
                "negative_prompt": negative_prompt,
                "config": Config(**config_kwargs),
            }

            # Qwen Edit uses image_paths instead of image_path in config
            # Check if it's Qwen Edit by checking the class name
            if "QwenImageEdit" in model_class.__name__:
                if image_paths:
                    generate_kwargs["image_paths"] = image_paths
                else:
                    generate_kwargs["image_paths"] = [str(image_path)]

            # Use a temporary directory for stepwise handler output
            import tempfile

            temp_dir = tempfile.mkdtemp()
            handler = StepwiseHandler(model=model, output_dir=temp_dir)
            CallbackRegistry.register_before_loop(handler)
            CallbackRegistry.register_in_loop(handler)
            CallbackRegistry.register_interrupt(handler)

            image = model.generate_image(**generate_kwargs)
            image.save(path=output_image_path, overwrite=True)

            # then
            model_name = "qwen edit"
            ImageCompare.check_images_close_enough(
                output_image_path,
                reference_image_path,
                f"Generated {model_name} image doesn't match reference image.",
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
