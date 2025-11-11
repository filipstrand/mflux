from pathlib import Path
from typing import Any, Type

from mflux.callbacks.callback_registry import CallbackRegistry
from mflux.callbacks.instances.stepwise_handler import StepwiseHandler
from mflux.config.config import Config
from mflux.config.model_config import ModelConfig

from .image_compare import check_images_close_enough


# Lazy import Flux1Kontext to avoid import errors when only running Qwen tests
def _get_flux1_kontext():
    from mflux.models.flux.variants.kontext.flux_kontext import Flux1Kontext

    return Flux1Kontext


# Lazy import QwenImageEditPlus to avoid import errors when only running regular edit tests
def _get_qwen_image_edit_plus():
    from mflux.models.qwen.variants.edit.qwen_image_edit_plus import QwenImageEditPlus

    return QwenImageEditPlus


class ImageGeneratorEditTestHelper:
    @staticmethod
    def assert_matches_reference_image(
        reference_image_path: str,
        output_image_path: str,
        model_class: Type[Any],  # Accepts QwenImageEdit or QwenImageEditPlus
        model_config: ModelConfig,
        steps: int,
        seed: int,
        height: int,
        width: int,
        prompt: str,
        image_path: str,
        guidance: float = 2.5,
        negative_prompt: str | None = None,
        quantize: int = 8,
    ):
        # resolve paths
        reference_image_path = ImageGeneratorEditTestHelper.resolve_path(reference_image_path)
        output_image_path = ImageGeneratorEditTestHelper.resolve_path(output_image_path)
        image_path = ImageGeneratorEditTestHelper.resolve_path(image_path)

        try:
            # given
            model_kwargs = {
                "quantize": quantize,
            }
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

            # Edit Plus uses image_paths instead of image_path in config
            # Check if it's Edit Plus by checking the class name
            if "Plus" in model_class.__name__:
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
            check_images_close_enough(
                output_image_path,
                reference_image_path,
                f"Generated {model_name} image doesn't match reference image.",
            )
        finally:
            # cleanup
            pass
            # if os.path.exists(output_image_path) and "MFLUX_PRESERVE_TEST_OUTPUT" not in os.environ:
            #     os.remove(output_image_path)

    @staticmethod
    def resolve_path(path) -> Path | None:
        if path is None:
            return None
        return Path(__file__).parent.parent.parent / "resources" / path
