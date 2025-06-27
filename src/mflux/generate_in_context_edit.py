from pathlib import Path

from PIL import Image

from mflux import Config, ModelConfig, StopImageGenerationException
from mflux.callbacks.callback_manager import CallbackManager
from mflux.community.in_context.flux_in_context_fill import Flux1InContextFill
from mflux.community.in_context.utils.in_context_loras import prepare_ic_edit_loras
from mflux.error.exceptions import PromptFileReadError
from mflux.ui import defaults as ui_defaults
from mflux.ui.cli.parsers import CommandLineParser
from mflux.ui.prompt_utils import get_effective_prompt


def main():
    # 0. Parse command line arguments
    parser = CommandLineParser(description="Generate images using in-context editing.")
    parser.add_general_arguments()
    parser.add_model_arguments(require_model_arg=False)
    parser.add_lora_arguments()
    parser.add_image_generator_arguments(supports_metadata_config=False, require_prompt=False)
    parser.add_in_context_edit_arguments()
    parser.add_in_context_arguments()
    parser.add_output_arguments()
    args = parser.parse_args()

    # 0. Default to a higher guidance value for fill related tasks.
    if args.guidance is None:
        args.guidance = ui_defaults.DEFAULT_DEV_FILL_GUIDANCE

    # Set sensible VAE tiling split for in-context generation (side-by-side images)
    if args.vae_tiling:
        args.vae_tiling_split = "vertical"

    # Auto-resize to optimal width for IC-Edit
    width, height = _resize_for_ic_edit_optimal_width(args)

    # 1. Load the model with IC-Edit LoRA
    flux = Flux1InContextFill(
        model_config=ModelConfig.dev_fill(),
        quantize=args.quantize,
        local_path=args.path,
        lora_paths=prepare_ic_edit_loras(args.lora_paths),
        lora_scales=args.lora_scales,
    )

    # 2. Register callbacks
    memory_saver = CallbackManager.register_callbacks(args=args, flux=flux)

    try:
        for seed in args.seed:
            # 3. Generate an image for each seed value
            image = flux.generate_image(
                seed=seed,
                prompt=_get_effective_ic_edit_prompt(args),
                left_image_path=args.reference_image,
                right_image_path=None,
                config=Config(
                    num_inference_steps=args.steps,
                    height=height,
                    width=width,
                    guidance=args.guidance,
                ),
            )

            # 4. Save the image(s)
            output_path = Path(args.output.format(seed=seed))
            image.get_right_half().save(path=output_path, export_json_metadata=args.metadata)
            if args.save_full_image:
                image.save(path=output_path.with_stem(output_path.stem + "_full"))

    except (StopImageGenerationException, PromptFileReadError) as exc:
        print(exc)
    finally:
        if memory_saver:
            print(memory_saver.memory_stats())


def _get_effective_ic_edit_prompt(args):
    if hasattr(args, "instruction") and args.instruction:
        return f"A diptych with two side-by-side images of the same scene. On the right, the scene is exactly the same as on the left but {args.instruction}"
    else:
        return get_effective_prompt(args)


def _resize_for_ic_edit_optimal_width(args):
    with Image.open(args.reference_image) as img:
        actual_width, actual_height = img.size
    aspect_ratio = actual_height / actual_width
    original_args_width = args.width
    original_args_height = args.height
    optimal_width = 512
    optimal_height = int(512 * aspect_ratio)
    optimal_height = (optimal_height // 8) * 8
    print(f"[INFO] IC-Edit LoRA trained on 512px width. Auto-resizing from actual image {actual_width}x{actual_height} to {optimal_width}x{optimal_height}")  # fmt:off
    print(f"[INFO] Aspect ratio maintained: {aspect_ratio:.3f}")
    if original_args_width != actual_width or original_args_height != actual_height:
        print(f"[INFO] Note: Command line args specified {original_args_width}x{original_args_height}, but using actual image dimensions for scaling")  # fmt:off
    return optimal_width, optimal_height


if __name__ == "__main__":
    main()
