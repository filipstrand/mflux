from argparse import Namespace

from mflux import Config, Flux1Controlnet, ModelConfig, StopImageGenerationException
from mflux.callbacks.callback_registry import CallbackRegistry
from mflux.callbacks.instances.battery_saver import BatterySaver
from mflux.callbacks.instances.canny_saver import CannyImageSaver
from mflux.callbacks.instances.memory_saver import MemorySaver
from mflux.callbacks.instances.stepwise_handler import StepwiseHandler
from mflux.error.exceptions import PromptFileReadError
from mflux.ui.cli.parsers import CommandLineParser
from mflux.ui.prompt_utils import get_effective_prompt


def main():
    # 0. Parse command line arguments
    parser = CommandLineParser(description="Generate an image based on a prompt and a controlnet reference image.")  # fmt: off
    parser.add_general_arguments()
    parser.add_model_arguments(require_model_arg=True)
    parser.add_lora_arguments()
    parser.add_image_generator_arguments(supports_metadata_config=False)
    parser.add_controlnet_arguments()
    parser.add_output_arguments()
    args = parser.parse_args()

    # 1. Load the model
    flux = Flux1Controlnet(
        model_config=_get_controlnet_model_config(args.model),
        quantize=args.quantize,
        local_path=args.path,
        lora_paths=args.lora_paths,
        lora_scales=args.lora_scales,
    )

    # 2. Register callbacks
    memory_saver = _register_callbacks(args=args, flux=flux)

    try:
        for seed in args.seed:
            # 3. Generate an image for each seed value
            image = flux.generate_image(
                seed=seed,
                prompt=get_effective_prompt(args),
                controlnet_image_path=args.controlnet_image_path,
                config=Config(
                    num_inference_steps=args.steps,
                    height=args.height,
                    width=args.width,
                    guidance=args.guidance,
                    controlnet_strength=args.controlnet_strength,
                ),
            )

            # 4. Save the image
            image.save(path=args.output.format(seed=seed), export_json_metadata=args.metadata)
    except (StopImageGenerationException, PromptFileReadError) as exc:
        print(exc)
    finally:
        if memory_saver:
            print(memory_saver.memory_stats())


def _get_controlnet_model_config(model_name: str) -> ModelConfig:
    if model_name == "schnell":
        return ModelConfig.schnell_controlnet_canny()
    return ModelConfig.dev_controlnet_canny()


def _register_callbacks(args: Namespace, flux: Flux1Controlnet) -> MemorySaver | None:
    # Battery saver
    battery_saver = BatterySaver(battery_percentage_stop_limit=args.battery_percentage_stop_limit)
    CallbackRegistry.register_before_loop(battery_saver)

    # VAE Tiling
    if args.vae_tiling:
        flux.vae.decoder.enable_tiling = True

    # Canny Image Saver
    if args.controlnet_save_canny:
        canny_image_saver = CannyImageSaver(path=args.output)
        CallbackRegistry.register_before_loop(canny_image_saver)

    # Stepwise Handler
    if args.stepwise_image_output_dir:
        handler = StepwiseHandler(flux=flux, output_dir=args.stepwise_image_output_dir)
        CallbackRegistry.register_before_loop(handler)
        CallbackRegistry.register_in_loop(handler)
        CallbackRegistry.register_interrupt(handler)

    # Memory Saver
    memory_saver = None
    if args.low_ram:
        memory_saver = MemorySaver(flux=flux, keep_transformer=len(args.seed) > 1)
        CallbackRegistry.register_before_loop(memory_saver)
        CallbackRegistry.register_in_loop(memory_saver)
        CallbackRegistry.register_after_loop(memory_saver)
    return memory_saver


if __name__ == "__main__":
    main()
