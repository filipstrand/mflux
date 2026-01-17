from mflux.callbacks.callback_manager import CallbackManager
from mflux.cli.parser.parsers import CommandLineParser
from mflux.models.common.config import ModelConfig
from mflux.models.flux2.latent_creator.flux2_latent_creator import Flux2LatentCreator
from mflux.models.flux2.variants import Flux2Klein
from mflux.utils.dimension_resolver import DimensionResolver
from mflux.utils.exceptions import PromptFileReadError, StopImageGenerationException
from mflux.utils.image_util import ImageUtil
from mflux.utils.prompt_util import PromptUtil


def main():
    # 0. Parse command line arguments
    parser = CommandLineParser(description="Generate an image using Flux2 Klein.")
    parser.add_general_arguments()
    parser.add_model_arguments(require_model_arg=False)
    parser.add_lora_arguments()
    parser.add_image_generator_arguments(supports_metadata_config=True, supports_dimension_scale_factor=True)
    parser.add_output_arguments()
    args = parser.parse_args()

    if getattr(args, "negative_prompt", ""):
        parser.error("--negative-prompt is not supported for FLUX.2. Focus on describing what you want.")

    if args.guidance is None:
        args.guidance = 1.0
    if args.guidance != 1.0:
        parser.error("--guidance is not supported for FLUX.2. Use --guidance 1.0.")

    model_name = args.model or "flux2-klein-4b"
    model = Flux2Klein(
        model_config=ModelConfig.from_name(model_name=model_name),
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
            reference_image_path=None,
        )

        for seed in args.seed:
            image = model.generate_image(
                seed=seed,
                prompt=PromptUtil.read_prompt(args),
                width=width,
                height=height,
                guidance=args.guidance,
                num_inference_steps=args.steps,
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


if __name__ == "__main__":
    main()
