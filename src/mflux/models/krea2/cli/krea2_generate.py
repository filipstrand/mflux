from mflux.callbacks.callback_manager import CallbackManager
from mflux.cli.defaults import defaults as ui_defaults
from mflux.cli.parser.parsers import CommandLineParser
from mflux.models.common.config import ModelConfig
from mflux.models.krea2.latent_creator import Krea2LatentCreator
from mflux.models.krea2.variants.txt2img.krea2 import Krea2Image
from mflux.utils.dimension_resolver import DimensionResolver
from mflux.utils.exceptions import PromptFileReadError, StopImageGenerationException
from mflux.utils.prompt_util import PromptUtil


def main():
    parser = CommandLineParser(description="Generate an image using Krea-2.")
    parser.add_general_arguments()
    parser.add_model_arguments(require_model_arg=False)
    parser.add_lora_arguments()
    parser.add_image_generator_arguments(supports_metadata_config=True, supports_dimension_scale_factor=True)
    parser.add_image_to_image_arguments(required=False)
    parser.add_output_arguments()
    args = parser.parse_args()

    if args.model is None:
        args.model = "krea2-turbo"
        args.model_path = None

    model_config = ModelConfig.from_name(model_name=args.model, base_model=args.base_model)
    guidance_was_provided = CommandLineParser._option_was_provided("--guidance")
    if not CommandLineParser._option_was_provided("--steps"):
        args.steps = ui_defaults.MODEL_INFERENCE_STEPS.get(
            args.model,
            ui_defaults.MODEL_INFERENCE_STEPS.get(model_config.aliases[0], 8),
        )
    if not CommandLineParser._option_was_provided("--scheduler"):
        args.scheduler = "flow_match_euler_discrete"
    if not model_config.supports_guidance:
        if guidance_was_provided and args.guidance != 0.0:
            parser.error("--guidance is only supported for Krea-2 Raw. Use --guidance 0.0 for Turbo.")
        args.guidance = 0.0
    elif args.guidance is None:
        args.guidance = 3.5

    krea2 = Krea2Image(
        model_config=model_config,
        quantize=args.quantize,
        model_path=args.model_path,
        lora_paths=args.lora_paths,
        lora_scales=args.lora_scales,
    )
    memory_saver = CallbackManager.register_callbacks(
        args=args,
        model=krea2,
        latent_creator=Krea2LatentCreator,
    )

    try:
        width, height = DimensionResolver.resolve(
            width=args.width,
            height=args.height,
            reference_image_path=args.image_path,
        )
        for seed in args.seed:
            image = krea2.generate_image(
                seed=seed,
                prompt=PromptUtil.read_prompt(args),
                negative_prompt=PromptUtil.read_negative_prompt(args),
                width=width,
                height=height,
                guidance=args.guidance,
                scheduler=args.scheduler,
                image_path=args.image_path,
                num_inference_steps=args.steps,
                image_strength=args.image_strength,
            )
            image.save(path=args.output.format(seed=seed), export_json_metadata=args.metadata)
    except (StopImageGenerationException, PromptFileReadError) as exc:
        print(exc)
    finally:
        if memory_saver:
            print(memory_saver.memory_stats())


if __name__ == "__main__":
    main()
