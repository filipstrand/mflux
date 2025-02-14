from mflux import Config, Flux1, ModelConfig, StopImageGenerationException
from mflux.callbacks.callback_registry import CallbackRegistry
from mflux.callbacks.instances.stepwise_handler import StepwiseHandler
from mflux.ui.cli.parsers import CommandLineParser

prompt = """
In this set of two images, a bold modern typeface with the brand name 'DEMA' is introduced and is shown on a company merchandise product photo; [IMAGE1] a simplistic black logo featuring a modern typeface with the brand name 'DEMA' on a bright light green/yellowish background; [IMAGE2] the design is printed on a green/yellowish hoodie as a company merchandise product photo with a plain white background.
"""


def main():
    # 0. Parse command line arguments
    parser = CommandLineParser(description="Generate an image based on a prompt.")
    parser.add_model_arguments(require_model_arg=False)
    parser.add_lora_arguments()
    parser.add_image_generator_arguments(supports_metadata_config=True)
    parser.add_image_to_image_arguments(required=False)
    parser.add_output_arguments()
    args = parser.parse_args()

    # 1. Load the model
    flux = Flux1(
        model_config=ModelConfig.dev(),
        quantize=8,
        local_path=args.path,
        lora_paths=["/Users/filipstrand/Desktop/visual-identity-design.safetensors"],
        lora_scales=[1.0],
    )

    # 2. Register the optional callbacks
    if True:
        handler = StepwiseHandler(flux=flux, output_dir="/Users/filipstrand/Desktop/ICLoRA")
        CallbackRegistry.register_before_loop(handler)
        CallbackRegistry.register_in_loop(handler)
        CallbackRegistry.register_interrupt(handler)

    try:
        for seed in args.seed:
            # 3. Generate an image for each seed value
            image = flux.generate_image(
                seed=42,
                prompt=prompt,
                config=Config(
                    num_inference_steps=20,
                    height=1024,
                    width=1024,
                    guidance=args.guidance,
                    init_image_path="/Users/filipstrand/Desktop/img1.png",
                    init_image_strength=0,
                ),
            )
            # 4. Save the image
            image.save(path=args.output.format(seed=seed), export_json_metadata=args.metadata)
    except StopImageGenerationException as stop_exc:
        print(stop_exc)


if __name__ == "__main__":
    main()
