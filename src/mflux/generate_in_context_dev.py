from pathlib import Path

from mflux import Config, StopImageGenerationException
from mflux.callbacks.callback_manager import CallbackManager
from mflux.community.in_context.flux_in_context_dev import Flux1InContextDev
from mflux.community.in_context.utils.in_context_loras import LORA_REPO_ID, get_lora_filename
from mflux.config.model_config import ModelConfig
from mflux.error.exceptions import PromptFileReadError
from mflux.ui import defaults as ui_defaults
from mflux.ui.cli.parsers import CommandLineParser
from mflux.ui.prompt_utils import get_effective_prompt


def main():
    # 0. Parse command line arguments
    parser = CommandLineParser(description="Generate an image using in-context LoRA with a reference image.")
    parser.add_general_arguments()
    parser.add_model_arguments(require_model_arg=False)
    parser.add_lora_arguments()
    parser.add_image_generator_arguments(supports_metadata_config=True)
    parser.add_image_to_image_arguments(required=True)
    parser.add_in_context_arguments()
    parser.add_output_arguments()
    args = parser.parse_args()

    # 0. Set default guidance value if not provided by user
    if args.guidance is None:
        args.guidance = ui_defaults.GUIDANCE_SCALE

    # Set sensible VAE tiling split for in-context generation (side-by-side images)
    if args.vae_tiling:
        args.vae_tiling_split = "vertical"

    # 1. Load the model
    flux = Flux1InContextDev(
        model_config=ModelConfig.dev(),
        quantize=args.quantize,
        lora_names=[get_lora_filename(args.lora_style)] if args.lora_style else None,
        lora_repo_id=LORA_REPO_ID if args.lora_style else None,
        lora_paths=args.lora_paths,
        lora_scales=args.lora_scales,
    )

    # 2. Register callbacks
    memory_saver = CallbackManager.register_callbacks(args=args, flux=flux)

    try:
        for seed in args.seed:
            # 3. Generate an image for each seed value
            image = flux.generate_image(
                seed=seed,
                prompt=get_effective_prompt(args),
                config=Config(
                    num_inference_steps=args.steps,
                    height=args.height,
                    width=args.width,
                    guidance=args.guidance,
                    image_path=args.image_path,
                ),
            )
            # 4. Save the image
            output_path = Path(args.output.format(seed=seed))
            image.get_right_half().save(path=output_path, export_json_metadata=args.metadata)
            if args.save_full_image:
                image.save(path=output_path.with_stem(output_path.stem + "_full"))

    except (StopImageGenerationException, PromptFileReadError) as exc:
        print(exc)
    finally:
        if memory_saver:
            print(memory_saver.memory_stats())


if __name__ == "__main__":
    main()
