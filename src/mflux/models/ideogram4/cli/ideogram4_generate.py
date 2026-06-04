from mflux.callbacks.callback_manager import CallbackManager
from mflux.cli.defaults import defaults as ui_defaults
from mflux.cli.parser.parsers import CommandLineParser
from mflux.models.common.config import ModelConfig
from mflux.models.ideogram4.config import is_ideogram4_alias
from mflux.models.ideogram4.latent_creator import Ideogram4LatentCreator
from mflux.models.ideogram4.scheduler import Ideogram4Scheduler
from mflux.models.ideogram4.variants.txt2img import Ideogram4
from mflux.utils.dimension_resolver import DimensionResolver
from mflux.utils.exceptions import PromptFileReadError, StopImageGenerationException
from mflux.utils.prompt_util import PromptUtil


def main():
    parser = CommandLineParser(description="Generate an image using Ideogram 4.")
    parser.add_general_arguments()
    parser.add_model_arguments(require_model_arg=False)
    parser.add_image_generator_arguments(supports_metadata_config=True)
    parser.add_output_arguments()
    parser.add_argument(
        "--preset",
        type=str,
        default=None,
        choices=sorted(Ideogram4Scheduler.PRESETS),
        help="Ideogram 4 sampler preset. Default is V4_DEFAULT_20.",
    )
    parser.add_argument(
        "--use-preset-steps",
        action="store_true",
        help="Use the selected preset's step count and guidance schedule.",
    )
    parser.add_argument(
        "--strict-caption-validation",
        action="store_true",
        help="Fail when an Ideogram 4 JSON caption has schema warnings.",
    )
    args = parser.parse_args()

    model_name = args.model or "ideogram4"
    use_builtin_model = is_ideogram4_alias(model_name)
    model_config = ModelConfig.from_name(model_name) if use_builtin_model else ModelConfig.ideogram4_fp8()
    model_path = None if use_builtin_model else args.model_path
    if not CommandLineParser._option_was_provided("--steps"):
        args.steps = ui_defaults.MODEL_INFERENCE_STEPS["ideogram4"]

    model = Ideogram4(
        model_config=model_config,
        quantize=args.quantize,
        model_path=model_path,
    )

    memory_saver = CallbackManager.register_callbacks(
        args=args,
        model=model,
        latent_creator=Ideogram4LatentCreator,
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
                preset=args.preset,
                use_preset_steps=args.use_preset_steps,
                strict_caption_validation=args.strict_caption_validation,
            )
            image.save(path=args.output.format(seed=seed), export_json_metadata=args.metadata)
    except (StopImageGenerationException, PromptFileReadError) as exc:
        print(exc)
    finally:
        if memory_saver:
            print(memory_saver.memory_stats())


if __name__ == "__main__":
    main()
