from mflux import Config, StopImageGenerationException
from mflux.callbacks.callback_registry import CallbackRegistry
from mflux.callbacks.instances.memory_saver import MemorySaver
from mflux.callbacks.instances.stepwise_handler import StepwiseHandler
from mflux.error.exceptions import PromptFileReadError
from mflux.flux_tools.fill.flux_fill import Flux1Fill
from mflux.ui.cli.parsers import CommandLineParser
from mflux.ui.prompt_utils import get_effective_prompt


def main():
    # 0. Parse command line arguments
    parser = CommandLineParser(description="Generate an image using the fill tool to complete masked areas.")
    parser.add_general_arguments()
    parser.add_model_arguments(require_model_arg=False)
    parser.add_lora_arguments()
    parser.add_image_generator_arguments(supports_metadata_config=False)
    parser.add_fill_arguments()
    parser.add_output_arguments()
    args = parser.parse_args()

    # 0. Default to a higher guidance value for fill related tasks.
    if args.guidance is None:
        args.guidance = 30

    # 1. Load the model
    flux = Flux1Fill(
        quantize=args.quantize,
        local_path=args.path,
        lora_paths=args.lora_paths,
        lora_scales=args.lora_scales,
    )

    # 2. Register the optional callbacks
    if args.stepwise_image_output_dir:
        handler = StepwiseHandler(flux=flux, output_dir=args.stepwise_image_output_dir)
        CallbackRegistry.register_before_loop(handler)
        CallbackRegistry.register_in_loop(handler)
        CallbackRegistry.register_interrupt(handler)

    memory_saver = None
    if args.low_ram:
        memory_saver = MemorySaver(flux=flux, keep_transformer=len(args.seed) > 1)
        CallbackRegistry.register_before_loop(memory_saver)
        CallbackRegistry.register_in_loop(memory_saver)
        CallbackRegistry.register_after_loop(memory_saver)

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
                    masked_image_path=args.masked_image_path,
                ),
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
