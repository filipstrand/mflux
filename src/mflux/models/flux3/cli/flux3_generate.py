from mflux.callbacks.callback_manager import CallbackManager
from mflux.cli.parser.parsers import CommandLineParser, lora_init_kwargs_from_args
from mflux.models.common.compute_precision import ComputePrecision
from mflux.models.common.config.model_config import AVAILABLE_MODELS
from mflux.models.common.resolution.config_resolution import ConfigResolution
from mflux.models.flux3.latent_creator.flux3_latent_creator import Flux3LatentCreator
from mflux.models.flux3.variants import Flux3Klein
from mflux.utils.dimension_resolver import DimensionResolver
from mflux.utils.exceptions import PromptFileReadError, StopImageGenerationException
from mflux.utils.prompt_util import PromptUtil

# The model this CLI runs when --model is omitted; the parser keys the --steps default (16) off it.
DEFAULT_MODEL = "flux3-image-klein"

# The entries this CLI equally serves: the undistilled Base checkpoint (50 steps, CFG, --guidance defaults to 3.5).
FAMILY_MODELS = tuple(key for key in AVAILABLE_MODELS if key.startswith("flux3-") and key != DEFAULT_MODEL)

IGNORED_OPTIONS = {
    "--scheduler": "FLUX 3 Image Klein always runs the flow-map schedule its checkpoints were trained with (Base uses the same schedule with CFG).",
    "--lora-style": "Named LoRA styles are only supported by the Flux in-context CLI; use --lora.",
}

CONDITIONAL_OPTIONS = {
    "--guidance": {
        "condition": "base (non-distilled) FLUX 3 Image Klein checkpoints",
        "reason": "the distilled checkpoint has no CFG branch, so the value is ignored with a warning.",
    },
}

REJECTED_OPTIONS = {
    "--negative-prompt": "FLUX 3 Image Klein takes no negative prompt; the CLI exits with an error when this is set.",
}


def build_parser() -> CommandLineParser:
    parser = CommandLineParser(description="Generate an image using FLUX 3 Image Klein.")
    parser.add_general_arguments()
    parser.add_model_arguments(require_model_arg=False, default_model=DEFAULT_MODEL)
    parser.add_image_generator_arguments(supports_metadata_config=True, supports_dimension_scale_factor=True)
    parser.add_compute_precision_arguments()
    parser.add_lora_arguments()
    parser.add_output_arguments()
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    # Keyed on the option, not the value: a metadata sidecar can restore a negative prompt nobody typed.
    if CommandLineParser._option_was_provided("--negative-prompt"):
        parser.error("--negative-prompt is not supported for FLUX 3 Image Klein. Focus on describing what you want.")

    model_config = ConfigResolution.resolve_restricted(
        args.model,
        DEFAULT_MODEL,
        model_path=args.model_path,
        extra_keys=FAMILY_MODELS,
        base_model=args.base_model,
    )

    CommandLineParser.warn_ignored_options(IGNORED_OPTIONS)
    if not model_config.supports_guidance:
        CommandLineParser.warn_ignored_options({"--guidance": CONDITIONAL_OPTIONS["--guidance"]["reason"]})
    model = Flux3Klein(
        model_config=model_config,
        quantize=args.quantize,
        model_path=args.model_path,
        **lora_init_kwargs_from_args(args),
        compute_precision=ComputePrecision.dtype_for(args.compute_precision),
    )

    memory_saver = CallbackManager.register_callbacks(
        args=args,
        model=model,
        latent_creator=Flux3LatentCreator,
    )

    try:
        width, height = DimensionResolver.resolve(width=args.width, height=args.height, reference_image_path=None)
        for seed in args.seed:
            image = model.generate_image(
                seed=seed,
                prompt=PromptUtil.read_prompt(args),
                width=width,
                height=height,
                num_inference_steps=args.steps,
                guidance=args.guidance,
            )
            image.save(path=args.output.format(seed=seed), export_json_metadata=args.metadata)
    except (StopImageGenerationException, PromptFileReadError) as exc:
        print(exc)
    finally:
        if memory_saver:
            print(memory_saver.memory_stats())


if __name__ == "__main__":
    main()
