from mflux import Config, Flux1Controlnet, ModelLookup, StopImageGenerationException
from mflux.callbacks.callback_registry import CallbackRegistry
from mflux.callbacks.instances.canny_saver import CannyImageSaver
from mflux.callbacks.instances.stepwise_handler import StepwiseHandler
from mflux.ui.cli.parsers import CommandLineParser


def main():
    # 0. Parse command line arguments
    parser = CommandLineParser(description="Generate an image based on a prompt and a controlnet reference image.")  # fmt: off
    parser.add_model_arguments(require_model_arg=True)
    parser.add_lora_arguments()
    parser.add_image_generator_arguments(supports_metadata_config=False)
    parser.add_controlnet_arguments()
    parser.add_output_arguments()
    args = parser.parse_args()

    # 1. Load the model
    flux = Flux1Controlnet(
        model_config=ModelLookup.from_name(model_name=args.model, base_model=args.base_model),
        quantize=args.quantize,
        local_path=args.path,
        lora_paths=args.lora_paths,
        lora_scales=args.lora_scales,
    )

    # 2. Register the optional callbacks
    if args.controlnet_save_canny:
        CallbackRegistry.register_before_loop(CannyImageSaver(path=args.output))
    if args.stepwise_image_output_dir:
        handler = StepwiseHandler(flux=flux, output_dir=args.stepwise_image_output_dir)
        CallbackRegistry.register_in_loop(handler)
        CallbackRegistry.register_interrupt(handler)

    try:
        for seed in args.seed:
            # 3. Generate an image for each seed value
            image = flux.generate_image(
                seed=seed,
                prompt=args.prompt,
                controlnet_image_path=args.controlnet_image_path,
                config=Config(
                    num_inference_steps=args.steps,
                    height=args.height,
                    width=args.width,
                    guidance=args.guidance,
                    controlnet_strength=args.controlnet_strength,
                ),
            )

            # 4. Save the image
            image.save(path=args.output.format(seed=seed), export_json_metadata=args.metadata)
    except StopImageGenerationException as stop_exc:
        print(stop_exc)


if __name__ == "__main__":
    main()
