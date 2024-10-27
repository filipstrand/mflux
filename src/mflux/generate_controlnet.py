import time
from pathlib import Path

from mflux import ConfigControlnet, Flux1Controlnet, ModelConfig, StopImageGenerationException
from mflux.ui.cli.parsers import CommandLineParser


def main():
    parser = CommandLineParser(description="Generate an image based on a prompt and a controlnet reference image.")  # fmt: off
    parser.add_model_arguments(require_model_arg=True)
    parser.add_lora_arguments()
    parser.add_image_generator_arguments(supports_metadata_config=False)
    parser.add_controlnet_arguments()
    parser.add_output_arguments()
    args = parser.parse_args()

    # Load the model
    flux = Flux1Controlnet(
        model_config=ModelConfig.from_alias(args.model),
        quantize=args.quantize,
        local_path=args.path,
        lora_paths=args.lora_paths,
        lora_scales=args.lora_scales,
    )

    try:
        # Generate an image
        image = flux.generate_image(
            seed=int(time.time()) if args.seed is None else args.seed,
            prompt=args.prompt,
            output=args.output,
            controlnet_image_path=args.controlnet_image_path,
            controlnet_save_canny=args.controlnet_save_canny,
            stepwise_output_dir=Path(args.stepwise_image_output_dir) if args.stepwise_image_output_dir else None,
            config=ConfigControlnet(
                num_inference_steps=args.steps,
                height=args.height,
                width=args.width,
                guidance=args.guidance,
                controlnet_strength=args.controlnet_strength,
            ),
        )

        # Save the image
        image.save(path=args.output, export_json_metadata=args.metadata)
    except StopImageGenerationException as stop_exc:
        print(stop_exc)


if __name__ == "__main__":
    main()
