"""
Chroma Image Generation CLI.

Command-line interface for generating images with the Chroma model.
"""

from mflux.callbacks.callback_manager import CallbackManager
from mflux.cli.parser.parsers import CommandLineParser
from mflux.models.chroma.variants.txt2img.chroma import Chroma
from mflux.models.common.config import ModelConfig
from mflux.models.flux.latent_creator.flux_latent_creator import FluxLatentCreator
from mflux.utils.dimension_resolver import DimensionResolver
from mflux.utils.exceptions import PromptFileReadError, StopImageGenerationException
from mflux.utils.prompt_util import PromptUtil


def main():
    """Main entry point for mflux-chroma-generate command."""
    # 0. Parse command line arguments
    parser = CommandLineParser(description="Generate an image with Chroma model.")
    parser.add_general_arguments()
    parser.add_model_arguments(require_model_arg=False)
    # No LoRA support in initial version
    # parser.add_lora_arguments()
    parser.add_image_generator_arguments(
        supports_metadata_config=True,
        supports_dimension_scale_factor=True,
    )
    parser.add_image_to_image_arguments(required=False)
    parser.add_output_arguments()
    args = parser.parse_args()

    # Set default values for Chroma
    if args.guidance is None:
        args.guidance = 4.0  # Chroma default

    if args.steps is None:
        args.steps = 40  # Chroma recommended steps

    if args.model is None:
        args.model = "chroma"

    # 1. Load the model
    chroma = Chroma(
        model_config=ModelConfig.from_name(model_name=args.model, base_model=None),
        quantize=args.quantize,
        model_path=args.model_path,
    )

    # 2. Register callbacks
    memory_saver = CallbackManager.register_callbacks(
        args=args,
        model=chroma,
        latent_creator=FluxLatentCreator,
    )

    try:
        # Resolve dimensions
        width, height = DimensionResolver.resolve(
            height=args.height,
            width=args.width,
            reference_image_path=args.image_path,
        )

        for seed in args.seed:
            # 3. Generate an image for each seed value
            image = chroma.generate_image(
                seed=seed,
                prompt=PromptUtil.read_prompt(args),
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
            image.save(
                path=args.output.format(seed=seed),
                export_json_metadata=args.metadata,
            )
    except (StopImageGenerationException, PromptFileReadError) as exc:
        print(exc)
    finally:
        if memory_saver:
            print(memory_saver.memory_stats())


if __name__ == "__main__":
    main()
