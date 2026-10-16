import re

from mflux.callbacks.callback_manager import CallbackManager
from mflux.cli.defaults import defaults as ui_defaults
from mflux.cli.parser.parsers import CommandLineParser, lora_init_kwargs_from_args
from mflux.models.common.config import ModelConfig
from mflux.models.flux.latent_creator.flux_latent_creator import FluxLatentCreator
from mflux.models.flux.variants.txt2img.flux import Flux1
from mflux.utils.dimension_resolver import DimensionResolver
from mflux.utils.exceptions import PromptFileReadError, StopImageGenerationException
from mflux.utils.prompt_util import PromptUtil

# Single source of truth for options this CLI accepts but cannot honour: the runtime
# warning and the mflux-capabilities dump both read it.
IGNORED_OPTIONS = {
    "--negative-prompt": "FLUX.1 uses distilled guidance and has no negative branch.",
}
CONDITIONAL_OPTIONS = {
    "--guidance": {
        "condition": "the resolved model supports guidance (dev does; schnell does not)",
        "reason": "schnell builds no guidance embedder, so the value has no path to affect the output.",
    },
}


def build_parser() -> CommandLineParser:
    parser = CommandLineParser(description="Generate an image based on a prompt.")
    parser.add_general_arguments()
    parser.add_model_arguments(require_model_arg=False)
    parser.add_lora_arguments()
    parser.add_image_generator_arguments(supports_metadata_config=True, supports_dimension_scale_factor=True)
    parser.add_image_to_image_arguments(required=False)
    parser.add_pid_decode_arguments()
    parser.add_output_arguments()
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    CommandLineParser.warn_ignored_options(IGNORED_OPTIONS)

    # Matches the aliases (flux3-...) and the official repo ids (black-forest-labs/flux-3-image-klein[-base]).
    # Only the last path component counts, so a FLUX.1 checkpoint under a flux3-named folder still loads. For a
    # path or repo id an explicit --base-model names the family, so only that base is checked then.
    model_name = None if args.model_path and args.base_model else args.model
    names = (model_name and model_name.rstrip("/").rsplit("/", 1)[-1], args.base_model)
    if any(name and re.search(r"flux[-_]?3(?!\d)", name.lower()) for name in names):
        parser.error("FLUX 3 Image Klein is not supported by mflux-generate. Use mflux-generate-flux3 instead.")
    if (args.base_model and "klein" in args.base_model.lower()) or (args.model and "flux2-klein" in args.model.lower()):
        parser.error("FLUX.2 Klein is not supported by mflux-generate. Use mflux-generate-flux2 instead.")

    # 0. Set default guidance value if not provided by user
    if args.guidance is None:
        args.guidance = ui_defaults.GUIDANCE_SCALE

    # 1. Load the model
    model_config = ModelConfig.from_name(model_name=args.model, base_model=args.base_model)
    if not model_config.supports_guidance:
        CommandLineParser.warn_ignored_options({"--guidance": CONDITIONAL_OPTIONS["--guidance"]["reason"]})

    flux = Flux1(
        model_config=model_config,
        quantize=args.quantize,
        model_path=args.model_path,
        **lora_init_kwargs_from_args(args),
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
                pid_decode=args.pid_decode,
                pid_degrade_sigma=args.pid_degrade_sigma,
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
