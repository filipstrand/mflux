import time
from pathlib import Path

from mflux import Config, Flux1, ModelConfig, StopImageGenerationException
from mflux.ui.cli.parsers import CommandLineParser


def main():
    # fmt: off
    parser = CommandLineParser(description="Generate an image based on a prompt.")
    parser.add_model_arguments()
    parser.add_lora_arguments()
    parser.add_image_generator_arguments()
    parser.add_output_arguments()
    args = parser.parse_args()

    # Load the model
    flux = Flux1(
        model_config=ModelConfig.from_alias(args.model),
        quantize=args.quantize,
        local_path=args.path,
        lora_paths=args.lora_paths,
        lora_scales=args.lora_scales,
    )

    try:
        image_gen_steps = flux.generate_image(
            seed=int(time.time()) if args.seed is None else args.seed,
            prompt=args.prompt,
            stepwise_output_dir=Path(args.stepwise_image_output_dir) if args.stepwise_image_output_dir else None,
            stepwise_yield=True,
            config=Config(
                num_inference_steps=args.steps,
                height=args.height,
                width=args.width,
                guidance=args.guidance,
            ),
        )
        while True:
            stepwise_output = next(image_gen_steps)
            print(f"Stepwise output: {stepwise_output}")
    except StopIteration as e:
        # Save the image returned after the stepwise iterations
        image = e.value
        print(f"Image Generator returned final image: {image}")
        image.save(path=args.output, export_json_metadata=args.metadata)
    except StopImageGenerationException as stop_exc:
        print(stop_exc)


if __name__ == "__main__":
    import logging

    logging.setLevel(logging.INFO)
    main()
