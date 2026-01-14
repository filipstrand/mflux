"""CLI for FLUX.2 image generation.

Usage:
    mflux-generate-flux2 --prompt "a beautiful sunset over mountains" --output sunset.png
    mflux-generate-flux2 --prompt "a cat" --quantize 8 --steps 30 --guidance 3.5
"""

from mflux.callbacks.callback_manager import CallbackManager
from mflux.cli.defaults import defaults as ui_defaults
from mflux.cli.parser.parsers import CommandLineParser
from mflux.models.common.config import ModelConfig
from mflux.models.flux2.latent_creator.flux2_latent_creator import Flux2LatentCreator
from mflux.models.flux2.variants.txt2img.flux2 import Flux2
from mflux.utils.dimension_resolver import DimensionResolver
from mflux.utils.exceptions import PromptFileReadError, StopImageGenerationException
from mflux.utils.prompt_util import PromptUtil


def validate_inputs(prompt, height, width, steps, guidance):
    """Validate generation inputs."""
    if not prompt or not prompt.strip():
        raise ValueError("Prompt cannot be empty")
    if height <= 0 or width <= 0:
        raise ValueError("Height and width must be positive")
    if height % 64 != 0 or width % 64 != 0:
        print(f"Warning: Height ({height}) and width ({width}) should be divisible by 64 for optimal results")
    if steps <= 0:
        raise ValueError("Steps must be positive")
    if guidance < 0:
        raise ValueError("Guidance must be non-negative")


def main():
    """Main entry point for FLUX.2 image generation CLI."""
    # 0. Parse command line arguments
    parser = CommandLineParser(description="Generate an image using FLUX.2 (32B parameter model).")
    parser.add_general_arguments()
    parser.add_model_arguments(require_model_arg=False)
    parser.add_lora_arguments()
    parser.add_image_generator_arguments(supports_metadata_config=True, supports_dimension_scale_factor=True)
    parser.add_image_to_image_arguments(required=False)
    parser.add_output_arguments()
    args = parser.parse_args()

    # Set FLUX.2 default guidance value if not provided
    if args.guidance is None:
        args.guidance = 3.5  # FLUX.2 default guidance scale

    # Set FLUX.2 default steps if not provided
    if args.steps is None:
        args.steps = 30  # FLUX.2 default steps

    # Get model config - use flux2 as default
    if args.model:
        model_config = ModelConfig.from_name(model_name=args.model, base_model=args.base_model)
    else:
        model_config = ModelConfig.flux2()

    # Validate inputs early
    # Note: We need to read the prompt first to validate it
    try:
        prompt = PromptUtil.read_prompt(args)
    except PromptFileReadError as exc:
        print(exc)
        return

    validate_inputs(prompt, args.height, args.width, args.steps, args.guidance)

    # 1. Load the model
    flux2 = Flux2(
        model_config=model_config,
        quantize=args.quantize,
        model_path=args.model_path,
        lora_paths=args.lora_paths,
        lora_scales=args.lora_scales,
    )

    # 2. Register callbacks
    memory_saver = CallbackManager.register_callbacks(
        args=args,
        model=flux2,
        latent_creator=Flux2LatentCreator,
    )

    try:
        # Resolve dimensions (supports ScaleFactor like "2x" when --image-path is provided)
        width, height = DimensionResolver.resolve(
            height=args.height,
            width=args.width,
            reference_image_path=args.image_path,
        )

        for seed in args.seed:
            # 3. Generate an image for each seed value
            image = flux2.generate_image(
                seed=seed,
                prompt=prompt,
                width=width,
                height=height,
                guidance=args.guidance,
                scheduler=args.scheduler,
                image_path=args.image_path,
                num_inference_steps=args.steps,
                image_strength=args.image_strength,
                negative_prompt=PromptUtil.read_negative_prompt(args),
            )
            # 4. Save the image
            image.save(path=args.output.format(seed=seed), export_json_metadata=args.metadata)
    except (StopImageGenerationException, PromptFileReadError) as exc:
        print(exc)
    finally:
        if memory_saver:
            print(memory_saver.memory_stats())


if __name__ == "__main__":
    main()
