"""CLI entry point for mflux-generate-zimage command."""

import gc

import mlx.core as mx

from mflux.callbacks.callback_manager import CallbackManager
from mflux.config.config import Config
from mflux.config.model_config import ModelConfig
from mflux.ui.cli.parsers import CommandLineParser
from mflux.ui.prompt_utils import PromptUtils
from mflux.utils.exceptions import PromptFileReadError, StopImageGenerationException
from mflux.zimage import ZImage


def main():
    # Parse command line arguments (using shared infrastructure)
    parser = CommandLineParser(description="Generate images with Z-Image-Turbo")
    parser.add_general_arguments()
    parser.add_model_arguments(require_model_arg=False)
    parser.set_defaults(model="zimage-turbo")  # Default for validation
    parser.add_image_generator_arguments(supports_metadata_config=True)
    parser.add_image_to_image_arguments()  # Add img2img support
    parser.add_output_arguments()
    args = parser.parse_args()

    # Z-Image Turbo has CFG baked in - force guidance to 0
    args.guidance = 0.0

    # Default steps for Turbo (if not specified)
    if args.steps is None:
        args.steps = 9

    # Load model
    zimage = ZImage(
        model_config=ModelConfig.zimage_turbo(),
        quantize=args.quantize,
        local_path=args.path,
    )

    # Register callbacks
    memory_saver = CallbackManager.register_callbacks(args=args, model=zimage)

    try:
        for seed in args.seed:
            image = zimage.generate_image(
                seed=seed,
                prompt=PromptUtils.get_effective_prompt(args),
                config=Config(
                    num_inference_steps=args.steps,
                    height=args.height,
                    width=args.width,
                    guidance=args.guidance,
                    image_path=args.image_path,
                    image_strength=args.image_strength,
                ),
            )
            image.save(path=args.output.format(seed=seed), export_json_metadata=args.metadata)

            # Clean up image reference to prevent retention during exit
            del image
    except (StopImageGenerationException, PromptFileReadError) as exc:
        print(exc)
    finally:
        if memory_saver:
            print(memory_saver.memory_stats())

        # Clean up model reference to release all weights
        del zimage

        # Force garbage collection and clear MLX cache before exit
        # This prevents the ~32GB memory spike that occurs during Python cleanup
        gc.collect()
        mx.clear_cache()


if __name__ == "__main__":
    main()
