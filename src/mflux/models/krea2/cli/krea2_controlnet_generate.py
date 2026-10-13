import warnings
from argparse import Namespace

from mflux.callbacks.callback_manager import CallbackManager
from mflux.cli.parser.parsers import CommandLineParser, lora_init_kwargs_from_args
from mflux.models.common.config.model_config import ModelConfig
from mflux.models.common.resolution.config_resolution import ConfigResolution
from mflux.models.krea2.latent_creator import Krea2LatentCreator
from mflux.models.krea2.variants.controlnet.krea2_depth import Krea2Depth
from mflux.utils.dimension_resolver import DimensionResolver
from mflux.utils.exceptions import ModelConfigError, PromptFileReadError, StopImageGenerationException
from mflux.utils.generated_image import GeneratedImage
from mflux.utils.prompt_util import PromptUtil

DEFAULT_MODEL = "krea-2-depth-controlnet"
# The same ControlNet on Krea 2 Raw.
FAMILY_MODELS = ("krea-2-raw-depth-controlnet",)
DEFAULT_GUIDANCE = 1.0


def build_parser() -> CommandLineParser:
    parser = CommandLineParser(description="Generate an image with Krea 2 and its depth ControlNet.")
    parser.add_general_arguments()
    parser.add_model_arguments(require_model_arg=False, default_model=DEFAULT_MODEL)
    parser.add_lora_arguments()
    parser.add_image_generator_arguments(supports_metadata_config=False, supports_dimension_scale_factor=True)
    parser.add_depth_arguments()
    parser._option_string_actions["--image-path"].help = "Source image. Depth Pro estimates its depth map."
    parser._option_string_actions[
        "--depth-image-path"
    ].help = "Ready-made depth map, near is white. Used instead of estimating one from --image-path."
    parser.add_argument(
        "--controlnet-path",
        type=str,
        default=None,
        help="Depth ControlNet checkpoint: a local .safetensors file or a HuggingFace repo. "
        "Defaults to Patil/Krea-2-depth-controlnet.",
    )
    parser.add_argument(
        "--controlnet-strength",
        type=float,
        default=1.0,
        help="Scale of the control deltas merged into the base weights at load (default 1.0).",
    )
    parser.add_output_arguments()
    return parser


class Krea2ControlnetCommand:
    latent_creator = Krea2LatentCreator

    @staticmethod
    def validate(args: Namespace) -> ModelConfig:
        # Weight-free: checked before the model loads, so a bad request fails fast.
        model_config = ConfigResolution.resolve_restricted(
            args.model, DEFAULT_MODEL, model_path=args.model_path, extra_keys=FAMILY_MODELS, base_model=args.base_model
        )
        if args.image_path is None and args.depth_image_path is None:
            raise ValueError("Give --depth-image-path, or --image-path to estimate the depth with Depth Pro.")
        return model_config

    @staticmethod
    def load(args: Namespace) -> Krea2Depth:
        return Krea2Depth(
            model_config=Krea2ControlnetCommand.validate(args),
            quantize=args.quantize,
            model_path=args.model_path,
            controlnet_path=args.controlnet_path,
            controlnet_strength=args.controlnet_strength,
            **lora_init_kwargs_from_args(args),
        )

    @staticmethod
    def generate(model: Krea2Depth, args: Namespace, seed: int, prompt: str) -> GeneratedImage:
        width, height = DimensionResolver.resolve(
            width=args.width,
            height=args.height,
            reference_image_path=args.image_path or args.depth_image_path,
        )
        return model.generate_image(
            seed=seed,
            prompt=prompt,
            num_inference_steps=args.steps,
            height=height,
            width=width,
            guidance=args.guidance if args.guidance is not None else DEFAULT_GUIDANCE,
            scheduler=args.scheduler,
            negative_prompt=args.negative_prompt,
            image_path=args.image_path,
            depth_image_path=args.depth_image_path,
        )


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    try:
        Krea2ControlnetCommand.validate(args)
    except ModelConfigError:
        raise
    except ValueError as exc:
        parser.error(str(exc))
    guidance = args.guidance if args.guidance is not None else DEFAULT_GUIDANCE
    if guidance == 1.0 and CommandLineParser._option_was_provided("--negative-prompt"):
        warnings.warn(
            "--negative-prompt is ignored at guidance 1.0; the encoder builds the unconditional branch "
            "only when guidance != 1.0.",
            stacklevel=2,
        )
    model = Krea2ControlnetCommand.load(args)
    memory_saver = CallbackManager.register_callbacks(
        args=args,
        model=model,
        latent_creator=Krea2ControlnetCommand.latent_creator,
        enable_depth_saver=True,
    )
    try:
        for seed in args.seed:
            image = Krea2ControlnetCommand.generate(model, args, seed, PromptUtil.read_prompt(args))
            image.save(path=args.output.format(seed=seed), export_json_metadata=args.metadata)
    except (StopImageGenerationException, PromptFileReadError) as exc:
        print(exc)
    finally:
        if memory_saver:
            print(memory_saver.memory_stats())


if __name__ == "__main__":
    main()
