import argparse
import time

from mflux import Flux1Controlnet, ConfigControlnet, ModelConfig


def main():
    # fmt: off
    parser = argparse.ArgumentParser(description="Generate an image based on a prompt.")
    parser.add_argument("--prompt", type=str, required=True, help="The textual description of the image to generate.")
    parser.add_argument("--controlnet-image-path", type=str, required=True, help="Local path of the image to use as input for controlnet.")
    parser.add_argument("--controlnet-strength", type=float, default=0.4, help="Controls how strongly the control image influences the output image. A value of 0.0 means no influence. (Default is 0.4)")
    parser.add_argument("--controlnet-save-canny", action="store_true", help="If set, save the Canny edge detection reference input image.")
    parser.add_argument("--output", type=str, default="image.png", help="The filename for the output image. Default is \"image.png\".")
    parser.add_argument("--model", "-m", type=str, required=True, choices=["dev", "schnell"], help="The model to use (\"schnell\" or \"dev\").")
    parser.add_argument("--seed", type=int, default=None, help="Entropy Seed (Default is time-based random-seed)")
    parser.add_argument("--height", type=int, default=1024, help="Image height (Default is 1024)")
    parser.add_argument("--width", type=int, default=1024, help="Image width (Default is 1024)")
    parser.add_argument("--steps", type=int, default=None, help="Inference Steps")
    parser.add_argument("--guidance", type=float, default=3.5, help="Guidance Scale (Default is 3.5)")
    parser.add_argument("--quantize",  "-q", type=int, choices=[4, 8], default=None, help="Quantize the model (4 or 8, Default is None)")
    parser.add_argument("--path", type=str, default=None, help="Local path for loading a model from disk")
    parser.add_argument("--lora-paths", type=str, nargs="*", default=None, help="Local safetensors for applying LORA from disk")
    parser.add_argument("--lora-scales", type=float, nargs="*", default=None, help="Scaling factor to adjust the impact of LoRA weights on the model. A value of 1.0 applies the LoRA weights as they are.")
    parser.add_argument("--metadata", action="store_true", help="Export image metadata as a JSON file.")
    # fmt: on

    args = parser.parse_args()

    if args.path and args.model is None:
        parser.error("--model must be specified when using --path")

    if args.steps is None:
        args.steps = 4 if args.model == "schnell" else 14

    # Load the model
    flux = Flux1Controlnet(
        model_config=ModelConfig.from_alias(args.model),
        quantize=args.quantize,
        local_path=args.path,
        lora_paths=args.lora_paths,
        lora_scales=args.lora_scales,
    )

    # Generate an image
    image = flux.generate_image(
        seed=int(time.time()) if args.seed is None else args.seed,
        prompt=args.prompt,
        output=args.output,
        controlnet_image_path=args.controlnet_image_path,
        controlnet_save_canny=args.controlnet_save_canny,
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


if __name__ == "__main__":
    main()
