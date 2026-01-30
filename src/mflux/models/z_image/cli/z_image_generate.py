from mflux.callbacks.callback_manager import CallbackManager
from mflux.cli.parser.parsers import CommandLineParser
from mflux.models.z_image.latent_creator import ZImageLatentCreator
from mflux.models.z_image.variants.txt2img.z_image import ZImage
from mflux.utils.dimension_resolver import DimensionResolver
from mflux.utils.exceptions import PromptFileReadError, StopImageGenerationException
from mflux.utils.prompt_util import PromptUtil


def main():
    # 0. Parse command line arguments
    parser = CommandLineParser(description="Generate an image using Z-Image base model with CFG support.")
    parser.add_general_arguments()
    parser.add_model_arguments(require_model_arg=False)
    parser.add_lora_arguments()
    parser.add_image_generator_arguments(supports_metadata_config=True, supports_dimension_scale_factor=True)
    parser.add_image_to_image_arguments(required=False)
    parser.add_output_arguments()

    # Add Z-Image specific arguments
    parser.parser.add_argument(
        "--negative-prompt",
        type=str,
        default="",
        help="Negative prompt describing what to avoid (default: empty)",
    )
    parser.parser.add_argument(
        "--guidance-scale",
        type=float,
        default=4.0,
        help="CFG guidance scale (recommended: 3.0-5.0, default: 4.0)",
    )
    parser.parser.add_argument(
        "--cfg-normalization",
        action="store_true",
        help="Enable CFG normalization (better for realism, default: False for stylism)",
    )

    args = parser.parse_args()

    # 1. Load the model
    model = ZImage(
        quantize=args.quantize,
        model_path=args.model_path,
        lora_paths=args.lora_paths,
        lora_scales=args.lora_scales,
    )

    # 2. Register callbacks
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
            # 3. Generate an image for each seed value
            image = model.generate_image(
                seed=seed,
                prompt=PromptUtil.read_prompt(args),
                negative_prompt=args.negative_prompt,
                width=width,
                height=height,
                guidance_scale=args.guidance_scale,
                cfg_normalization=args.cfg_normalization,
                image_path=args.image_path,
                num_inference_steps=args.steps,
                image_strength=args.image_strength,
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
