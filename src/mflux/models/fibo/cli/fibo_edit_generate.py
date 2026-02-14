from pathlib import Path

from mflux.callbacks.callback_manager import CallbackManager
from mflux.cli.defaults import defaults as ui_defaults
from mflux.cli.parser.parsers import CommandLineParser
from mflux.models.fibo.latent_creator.fibo_latent_creator import FiboLatentCreator
from mflux.models.fibo.variants.edit.fibo_edit import FIBOEdit
from mflux.models.fibo.variants.edit.util import FiboEditUtil
from mflux.models.fibo.variants.txt2img.util import FiboUtil
from mflux.utils.dimension_resolver import DimensionResolver
from mflux.utils.exceptions import PromptFileReadError, StopImageGenerationException
from mflux.utils.prompt_util import PromptUtil


def main():
    parser = CommandLineParser(description="Generate an edited image using Bria FIBO Edit.")
    parser.add_general_arguments()
    parser.add_model_arguments(require_model_arg=False)
    parser.add_lora_arguments()
    parser.add_image_generator_arguments(supports_metadata_config=True, supports_dimension_scale_factor=True)
    parser.add_argument("--image-path", type=Path, required=True, help="Local path to source image for editing.")
    parser.add_argument("--mask-path", type=Path, default=None, help="Optional mask image path for localized edits.")
    parser.add_argument(
        "--edit-instruction",
        type=str,
        default=None,
        help="Optional edit instruction. Used when prompt JSON does not already include `edit_instruction`.",
    )
    parser.add_output_arguments()
    args = parser.parse_args()

    if args.guidance is None:
        args.guidance = ui_defaults.GUIDANCE_SCALE

    json_prompt = FiboUtil.get_json_prompt(args, quantize=args.quantize)
    json_prompt = FiboEditUtil.ensure_edit_instruction(json_prompt, edit_instruction=args.edit_instruction)

    fibo_edit = FIBOEdit(
        quantize=args.quantize,
        model_path=args.model_path,
        lora_paths=args.lora_paths,
        lora_scales=args.lora_scales,
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
