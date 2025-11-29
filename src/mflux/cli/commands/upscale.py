import sys

from mflux.callbacks.callback_manager import CallbackManager
from mflux.cli.defaults import defaults as ui_defaults
from mflux.cli.parser.parsers import CommandLineParser
from mflux.config.config import Config
from mflux.config.model_config import ModelConfig
from mflux.models.flux.latent_creator.flux_latent_creator import FluxLatentCreator
from mflux.models.flux.variants.controlnet.flux_controlnet import Flux1Controlnet
from mflux.utils.dimension_resolver import DimensionResolver
from mflux.utils.exceptions import PromptFileReadError, StopImageGenerationException
from mflux.utils.prompt_util import PromptUtil


def main():
    # 0. Parse command line arguments
    parser = CommandLineParser(description="Upscale an image.")
    parser.add_general_arguments()
    parser.add_model_arguments(require_model_arg=False)
    parser.add_lora_arguments()
    parser.add_image_generator_arguments(supports_metadata_config=False, supports_dimension_scale_factor=True)
    parser.add_controlnet_arguments(require_image=True)
    parser.add_output_arguments()
    args = parser.parse_args()

    # 1. Load the model
    flux = Flux1Controlnet(
        model_config=ModelConfig.dev_controlnet_upscaler(),
        quantize=args.quantize,
        local_path=args.path,
        lora_paths=args.lora_paths,
        lora_scales=args.lora_scales,
    )

    # 2. Register the optional callbacks
    memory_saver = CallbackManager.register_callbacks(args=args, model=flux, latent_creator=FluxLatentCreator)

    try:
        # Calculate output dimensions and handle safety warnings
        width, height = _calculate_output_dimensions(args)

        for seed in args.seed:
            # 3. Generate an upscaled image for each seed value
            image = flux.generate_image(
                seed=seed,
                prompt=PromptUtil.get_effective_prompt(args),
                controlnet_image_path=args.controlnet_image_path,
                config=Config(
                    num_inference_steps=args.steps,
                    height=height,
                    width=width,
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


def _calculate_output_dimensions(args) -> tuple[int, int]:
    # Resolve dimensions using the controlnet image as reference
    output_width, output_height = DimensionResolver.resolve(
        height=args.height,
        width=args.width,
        reference_image_path=args.controlnet_image_path,
    )

    # Check if dimensions exceed safe limits
    total_pixels = output_height * output_width

    if total_pixels > ui_defaults.MAX_PIXELS_WARNING_THRESHOLD:
        print(
            f"⚠️ WARNING: The requested dimensions {output_width}x{output_height} "
            f"({total_pixels:,} pixels) exceed max recommended ({ui_defaults.MAX_PIXELS_WARNING_THRESHOLD:,} pixels)."
        )
        print("This generation is likely to exceed the capabilities of this computer and may:")
        print("  ⏳ Take a very long time to complete")
        print("  🔥 Run out of memory")
        print("  💥 Cause the program and your Mac to crash")

        user_input = input("\nPress Enter to continue at your own risk, or type 'n' to cancel: ")
        if user_input.lower() in ["n", "no"]:
            print("🛑 Generation cancelled by user.")
            sys.exit(1)

    return output_width, output_height


if __name__ == "__main__":
    main()
