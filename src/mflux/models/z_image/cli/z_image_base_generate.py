"""CLI for Z-Image-Base inference with CFG support.

Z-Image-Base differs from Z-Image-Turbo:
- Uses CFG (guidance_scale 3.0-5.0)
- Supports negative prompts
- Requires more steps (50 default vs 4 for Turbo)
- Provides higher quality and more control

Usage:
    mflux-generate-z-image-base --prompt "a cat" --guidance 3.5 --steps 50
    mflux-generate-z-image-base --prompt "a cat" --negative-prompt "blurry, low quality"
"""

from mflux.callbacks.callback_manager import CallbackManager
from mflux.cli.parser.parsers import CommandLineParser
from mflux.models.z_image.latent_creator import ZImageLatentCreator
from mflux.models.z_image.variants.training.z_image_base import ZImageBase
from mflux.utils.dimension_resolver import DimensionResolver
from mflux.utils.exceptions import PromptFileReadError, StopImageGenerationException
from mflux.utils.prompt_util import PromptUtil


def main():
    # Parse command line arguments
    parser = CommandLineParser(description="Generate an image using Z-Image-Base with CFG support.")
    parser.add_general_arguments()
    parser.add_model_arguments(require_model_arg=False)
    parser.add_lora_arguments()
    parser.add_image_generator_arguments(supports_metadata_config=True, supports_dimension_scale_factor=True)
    parser.add_image_to_image_arguments(required=False)
    parser.add_output_arguments()
    args = parser.parse_args()

    # Set default steps and guidance for Z-Image-Base
    if args.steps is None:
        args.steps = 50
    if args.guidance is None:
        args.guidance = 3.5

    # Load the model
    model = ZImageBase(
        quantize=args.quantize,
        model_path=args.model_path,
        lora_paths=args.lora_paths,
        lora_scales=args.lora_scales,
    )

    # Register callbacks
    memory_saver = CallbackManager.register_callbacks(
        args=args,
        model=model,
        latent_creator=ZImageLatentCreator,
    )

    try:
        # Resolve dimensions
        width, height = DimensionResolver.resolve(
            width=args.width,
            height=args.height,
            reference_image_path=args.image_path,
        )

        for seed in args.seed:
            # Generate image with CFG support
            image = model.generate_image(
                seed=seed,
                prompt=PromptUtil.read_prompt(args),
                negative_prompt=getattr(args, "negative_prompt", ""),
                width=width,
                height=height,
                guidance_scale=args.guidance,
                image_path=args.image_path,
                num_inference_steps=args.steps,
                image_strength=args.image_strength,
            )
            # Save the image
            image.save(path=args.output.format(seed=seed), export_json_metadata=args.metadata)

    except (StopImageGenerationException, PromptFileReadError) as exc:
        print(exc)
    finally:
        if memory_saver:
            print(memory_saver.memory_stats())


if __name__ == "__main__":
    main()
