from pathlib import Path
from typing import Type, Union

from mflux.callbacks.callback_registry import CallbackRegistry
from mflux.callbacks.instances.stepwise_handler import StepwiseHandler
from mflux.config.config import Config
from mflux.config.model_config import ModelConfig
from mflux.models.flux.variants.kontext.flux_kontext import Flux1Kontext
from mflux.models.qwen.variants.edit.qwen_image_edit import QwenImageEdit

from .image_compare import check_images_close_enough


class ImageGeneratorEditTestHelper:
    @staticmethod
    def assert_matches_reference_image(
        reference_image_path: str,
        output_image_path: str,
        model_class: Type[Union[Flux1Kontext, QwenImageEdit]],
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
            }
            generate_kwargs = {
                "seed": seed,
                "prompt": prompt,
                "negative_prompt": negative_prompt,
                "config": Config(**config_kwargs),
            }

            handler = StepwiseHandler(model=model, output_dir="/Users/filipstrand/Desktop/aaaa")
            CallbackRegistry.register_before_loop(handler)
            CallbackRegistry.register_in_loop(handler)
            CallbackRegistry.register_interrupt(handler)

            image = model.generate_image(**generate_kwargs)
            image.save(path=output_image_path, overwrite=True)

            # then
            model_name = "kontext" if model_class == Flux1Kontext else "qwen edit"
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
