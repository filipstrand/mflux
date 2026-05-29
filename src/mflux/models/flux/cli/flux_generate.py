from mflux.callbacks.callback_manager import CallbackManager
from mflux.cli.defaults import defaults as ui_defaults
from mflux.cli.parser.parsers import CommandLineParser
from mflux.models.common.config import ModelConfig
from mflux.models.flux.latent_creator.flux_latent_creator import FluxLatentCreator
from mflux.models.flux.variants.txt2img.flux import Flux1
from mflux.utils.dimension_resolver import DimensionResolver
from mflux.utils.exceptions import PromptFileReadError, StopImageGenerationException
from mflux.utils.prompt_util import PromptUtil


def main():
    # 0. Parse command line arguments
    parser = CommandLineParser(description="Generate an image based on a prompt.")
    parser.add_general_arguments()
    parser.add_model_arguments(require_model_arg=False)
    parser.add_lora_arguments()
    parser.add_image_generator_arguments(supports_metadata_config=True, supports_dimension_scale_factor=True)
    parser.add_image_to_image_arguments(required=False)
    parser.add_output_arguments()
    args = parser.parse_args()

    # Resolve model config early so we can dispatch on model family
    model_config = ModelConfig.from_name(model_name=args.model, base_model=args.base_model)

    # Redirect Klein (FLUX.2) models to the dedicated Flux2Klein implementation
    if model_config.is_klein():
        _run_klein(args, parser, model_config)
        return

    # 0. Set default guidance value if not provided by user
    if args.guidance is None:
        args.guidance = ui_defaults.GUIDANCE_SCALE

    # 1. Load the model
    flux = Flux1(
        model_config=model_config,
        quantize=args.quantize,
        model_path=args.model_path,
        lora_paths=args.lora_paths,
        lora_scales=args.lora_scales,
    )

    # 2. Register callbacks
    memory_saver = CallbackManager.register_callbacks(
        args=args,
        model=flux,
        latent_creator=FluxLatentCreator,
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
            image = flux.generate_image(
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
            image.save(path=args.output.format(seed=seed), export_json_metadata=args.metadata)
    except (StopImageGenerationException, PromptFileReadError) as exc:
        print(exc)
    finally:
        if memory_saver:
            print(memory_saver.memory_stats())


if __name__ == "__main__":
    main()


def _run_klein(args, parser, model_config: ModelConfig) -> None:
    """Delegate Klein models to the Flux2Klein implementation."""
    from mflux.models.flux2.latent_creator.flux2_latent_creator import Flux2LatentCreator
    from mflux.models.flux2.variants import Flux2Klein
    from mflux.utils.image_util import ImageUtil

    if args.guidance is None:
        args.guidance = 1.0
    is_distilled = "base" not in model_config.model_name.lower()
    if args.guidance != 1.0 and is_distilled:
        parser.error("--guidance is only supported for FLUX.2 base models. Use --guidance 1.0.")

    model = Flux2Klein(
        model_config=model_config,
        quantize=args.quantize,
        model_path=args.model_path,
        lora_paths=args.lora_paths,
        lora_scales=args.lora_scales,
    )

    memory_saver = CallbackManager.register_callbacks(
        args=args,
        model=model,
        latent_creator=Flux2LatentCreator,
    )

    try:
        width, height = DimensionResolver.resolve(
            width=args.width,
            height=args.height,
            reference_image_path=args.image_path,
        )
        for seed in args.seed:
            image = model.generate_image(
                seed=seed,
                prompt=PromptUtil.read_prompt(args),
                width=width,
                height=height,
                guidance=args.guidance,
                image_path=args.image_path,
                num_inference_steps=args.steps,
                image_strength=args.image_strength,
                scheduler="flow_match_euler_discrete",
            )
            ImageUtil.save_image(
                image=image,
                path=args.output.format(seed=seed),
                export_json_metadata=args.metadata,
            )
    except (StopImageGenerationException, PromptFileReadError) as exc:
        print(exc)
    finally:
        if memory_saver:
            print(memory_saver.memory_stats())
