from pathlib import Path

from mflux.callbacks.callback_manager import CallbackManager
from mflux.config.config import Config
from mflux.error.exceptions import PromptFileReadError, StopImageGenerationException
from mflux.models.qwen.variants.edit.qwen_image_edit import QwenImageEdit
from mflux.ui import defaults as ui_defaults
from mflux.ui.cli.parsers import CommandLineParser
from mflux.ui.prompt_utils import get_effective_negative_prompt, get_effective_prompt


def main():
    # 0. Parse command line arguments
    parser = CommandLineParser(description="Generate an image using Qwen Image Edit with image conditioning.")
    parser.add_general_arguments()
    parser.add_model_arguments(require_model_arg=False)
    parser.add_lora_arguments()
    parser.add_image_generator_arguments(supports_metadata_config=True)
    parser.add_image_to_image_arguments(required=True)
    parser.add_argument(
        "--image-paths",
        type=Path,
        nargs="+",
        default=None,
        help="Local paths to multiple init images. If not provided, uses --image-path as single image.",
    )
    parser.add_output_arguments()
    args = parser.parse_args()

    # 0. Set default guidance value if not provided by user
    if args.guidance is None:
        args.guidance = ui_defaults.GUIDANCE_SCALE_KONTEXT

    # 1. Load the model
    qwen = QwenImageEdit(
        quantize=args.quantize,
        local_path=args.path,
        lora_paths=args.lora_paths,
        lora_scales=args.lora_scales,
    )

    # 2. Register callbacks
    memory_saver = CallbackManager.register_callbacks(args=args, model=qwen)

    try:
        for seed in args.seed:
            # 3. Prepare image paths: use --image-paths if provided, otherwise fallback to --image-path
            image_paths = None
            if args.image_paths:
                image_paths = [str(p) for p in args.image_paths]
            elif args.image_path:
                image_paths = [str(args.image_path)]

            # 4. Generate an image for each seed value
            image = qwen.generate_image(
                seed=seed,
                prompt=get_effective_prompt(args),
                config=Config(
                    num_inference_steps=args.steps,
                    height=args.height,
                    width=args.width,
                    guidance=args.guidance,
                    image_path=args.image_path,
                ),
                negative_prompt=get_effective_negative_prompt(args),
                image_paths=image_paths,
            )

            # 5. Save the image
            output_path = Path(args.output.format(seed=seed))
            image.save(path=output_path, export_json_metadata=args.metadata)

    except (StopImageGenerationException, PromptFileReadError) as exc:
        print(exc)
    finally:
        if memory_saver:
            print(memory_saver.memory_stats())


if __name__ == "__main__":
    main()
