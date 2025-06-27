import sys

import PIL.Image

from mflux import Config, Flux1Controlnet, ModelConfig, StopImageGenerationException
from mflux.callbacks.callback_manager import CallbackManager
from mflux.error.exceptions import PromptFileReadError
from mflux.ui import defaults as ui_defaults
from mflux.ui.cli.parsers import CommandLineParser
from mflux.ui.prompt_utils import get_effective_prompt
from mflux.ui.scale_factor import ScaleFactor


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
    memory_saver = CallbackManager.register_callbacks(args=args, flux=flux)

    try:
        # Calculate output dimensions and handle safety warnings
        width, height = _calculate_output_dimensions(args)

        for seed in args.seed:
            # 3. Generate an upscaled image for each seed value
            image = flux.generate_image(
                seed=seed,
                prompt=get_effective_prompt(args),
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
    """Calculate output dimensions from args, handling scale factors and safety warnings."""
    # Image.open is lazy/efficient, just need the dimension metadata
    orig_image = PIL.Image.open(args.controlnet_image_path)
    output_width, output_height = orig_image.size

    if isinstance(args.height, ScaleFactor):
        output_height: int = args.height.get_scaled_value(orig_image.height)  # type: ignore

    else:
        output_height = args.height  # type: ignore

    if isinstance(args.width, ScaleFactor):
        output_width: int = args.width.get_scaled_value(orig_image.width)  # type: ignore

    else:
        output_width = args.width  # type: ignore

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
