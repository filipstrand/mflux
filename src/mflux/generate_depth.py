from argparse import Namespace

from mflux import Config, StopImageGenerationException
from mflux.callbacks.callback_registry import CallbackRegistry
from mflux.callbacks.instances.battery_saver import BatterySaver
from mflux.callbacks.instances.depth_saver import DepthImageSaver
from mflux.callbacks.instances.memory_saver import MemorySaver
from mflux.callbacks.instances.stepwise_handler import StepwiseHandler
from mflux.error.exceptions import PromptFileReadError
from mflux.flux_tools.depth.flux_depth import Flux1Depth
from mflux.ui.cli.parsers import CommandLineParser
from mflux.ui.prompt_utils import get_effective_prompt


def main():
    # 0. Parse command line arguments
    parser = CommandLineParser(description="Generate an image using the depth tool.")
    parser.add_general_arguments()
    parser.add_model_arguments(require_model_arg=False)
    parser.add_lora_arguments()
    parser.add_image_generator_arguments(supports_metadata_config=False)
    parser.add_depth_arguments()
    parser.add_output_arguments()
    args = parser.parse_args()

    # 0. Default to a medium guidance value for depth related tasks.
    if args.guidance is None:
        args.guidance = 10

    # 1. Load the model
    flux = Flux1Depth(
        quantize=args.quantize,
        local_path=args.path,
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
                    depth_image_path=args.depth_image_path,
                ),
            )

            # 4. Save the image
            image.save(path=args.output.format(seed=seed), export_json_metadata=args.metadata)
    except (StopImageGenerationException, PromptFileReadError) as exc:
        print(exc)
    finally:
        if memory_saver:
            print(memory_saver.memory_stats())


def _register_callbacks(args: Namespace, flux: Flux1Depth) -> MemorySaver | None:
    # Battery saver
    battery_saver = BatterySaver(battery_percentage_stop_limit=args.battery_percentage_stop_limit)
    CallbackRegistry.register_before_loop(battery_saver)

    # Depth Image Saver
    if args.save_depth_map:
        depth_image_saver = DepthImageSaver(path=args.output)
        CallbackRegistry.register_before_loop(depth_image_saver)

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
