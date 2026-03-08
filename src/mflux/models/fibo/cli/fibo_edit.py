from pathlib import Path

from mflux.callbacks.callback_manager import CallbackManager
from mflux.cli.defaults import defaults as ui_defaults
from mflux.cli.parser.parsers import CommandLineParser
from mflux.models.common.config.model_config import ModelConfig
from mflux.models.fibo.latent_creator.fibo_latent_creator import FiboLatentCreator
from mflux.models.fibo.variants.edit.fibo_edit import FIBOEdit
from mflux.models.fibo.variants.edit.util import FiboEditUtil
from mflux.utils.dimension_resolver import DimensionResolver
from mflux.utils.exceptions import ModelConfigError, PromptFileReadError, StopImageGenerationException
from mflux.utils.prompt_util import PromptUtil


def _resolve_fibo_edit_model_config(parser: CommandLineParser, args) -> ModelConfig:
    try:
        model_config = ModelConfig.from_name(model_name=args.model, base_model=args.base_model)
    except ModelConfigError as exc:
        parser.error(str(exc))

    if "fibo-edit" not in model_config.aliases:
        parser.error(
            "mflux-generate-fibo-edit requires a FIBO Edit model; metadata or --model resolved to an incompatible model."
        )

    return model_config


def main():
    parser = CommandLineParser(description="Generate an edited image using Bria FIBO Edit.")
    parser.add_general_arguments()
    parser.add_model_arguments(require_model_arg=False)
    parser.set_defaults(model="fibo-edit")
    parser.add_lora_arguments()
    parser.add_image_generator_arguments(
        supports_metadata_config=True,
        require_prompt=False,
        supports_dimension_scale_factor=True,
    )
    parser.add_argument("--image-path", type=Path, required=False, help="Local path to source image for editing.")
    parser.add_argument("--mask-path", type=Path, default=None, help="Optional mask image path for localized edits.")
    parser.add_output_arguments()
    args = parser.parse_args()

    if args.image_path is None:
        parser.error("--image-path is required, or 'image_path' must be specified in the config file.")

    if args.guidance is None:
        args.guidance = ui_defaults.GUIDANCE_SCALE

    model_config = _resolve_fibo_edit_model_config(parser, args)
    json_prompt = FiboEditUtil.get_json_prompt_for_edit(args, quantize=args.quantize)

    fibo_edit = FIBOEdit(
        quantize=args.quantize,
        model_path=args.model_path,
        lora_paths=args.lora_paths,
        lora_scales=args.lora_scales,
        model_config=model_config,
    )

    memory_saver = CallbackManager.register_callbacks(
        args=args,
        model=fibo_edit,
        latent_creator=FiboLatentCreator,
    )

    try:
        width, height = DimensionResolver.resolve(
            width=args.width,
            height=args.height,
            reference_image_path=args.image_path,
        )
        for seed in args.seed:
            image = fibo_edit.generate_image(
                seed=seed,
                prompt=json_prompt,
                image_path=args.image_path,
                mask_path=args.mask_path,
                width=width,
                height=height,
                guidance=args.guidance,
                num_inference_steps=args.steps,
                scheduler="flow_match_euler_discrete",
                negative_prompt=PromptUtil.read_negative_prompt(args),
            )
            image.save(path=args.output.format(seed=seed), export_json_metadata=args.metadata)
    except (StopImageGenerationException, PromptFileReadError, ValueError) as exc:
        print(exc)
    finally:
        if memory_saver:
            print(memory_saver.memory_stats())


if __name__ == "__main__":
    main()
