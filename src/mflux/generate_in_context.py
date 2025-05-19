from argparse import Namespace
from pathlib import Path

from mflux import Config, StopImageGenerationException
from mflux.callbacks.callback_registry import CallbackRegistry
from mflux.callbacks.instances.battery_saver import BatterySaver
from mflux.callbacks.instances.memory_saver import MemorySaver
from mflux.callbacks.instances.stepwise_handler import StepwiseHandler
from mflux.community.in_context_lora.flux_in_context_lora import Flux1InContextLoRA
from mflux.community.in_context_lora.in_context_loras import LORA_REPO_ID, get_lora_filename
from mflux.config.model_config import ModelConfig
from mflux.error.exceptions import PromptFileReadError
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
    parser.add_output_arguments()
    parser.add_argument(
        "--save-full-image",
        action="store_true",
        default=False,
        help="Additionally, save the full image containing the reference image. Useful for verifying the in-context usage of the reference image.",
    )
    args = parser.parse_args()

    # 1. Load the model
    flux = Flux1InContextLoRA(
        model_config=ModelConfig.dev(),
        quantize=args.quantize,
        lora_names=[get_lora_filename(args.lora_style)] if args.lora_style else None,
        lora_repo_id=LORA_REPO_ID if args.lora_style else None,
        lora_paths=args.lora_paths,
        lora_scales=args.lora_scales,
    )

    # 2. Register callbacks
    memory_saver = _register_callbacks(args=args, flux=flux)

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


def _register_callbacks(args: Namespace, flux: Flux1InContextLoRA) -> MemorySaver | None:
    # Battery saver
    battery_saver = BatterySaver(battery_percentage_stop_limit=args.battery_percentage_stop_limit)
    CallbackRegistry.register_before_loop(battery_saver)

    # Stepwise Handler
    if args.stepwise_image_output_dir:
        handler = StepwiseHandler(flux=flux, output_dir=args.stepwise_image_output_dir)
        CallbackRegistry.register_before_loop(handler)
        CallbackRegistry.register_in_loop(handler)
        CallbackRegistry.register_interrupt(handler)

    # Memory Saver
    memory_saver = None
    if args.low_ram:
        memory_saver = MemorySaver(flux=flux, keep_transformer=len(args.seed) > 1)
        CallbackRegistry.register_before_loop(memory_saver)
        CallbackRegistry.register_in_loop(memory_saver)
        CallbackRegistry.register_after_loop(memory_saver)
    return memory_saver


if __name__ == "__main__":
    main()
