"""
CLI for Hunyuan-DiT text-to-image generation.

Usage:
    mflux-generate-hunyuan --prompt "A beautiful sunset" --steps 50 --output sunset.png
"""

import argparse
import sys

from mflux.models.hunyuan.variants.txt2img.hunyuan import Hunyuan


def validate_inputs(prompt, height, width, steps, guidance):
    """Validate generation inputs."""
    if not prompt or not prompt.strip():
        raise ValueError("Prompt cannot be empty")
    if height <= 0 or width <= 0:
        raise ValueError("Height and width must be positive")
    if height % 64 != 0 or width % 64 != 0:
        print(f"Warning: Height ({height}) and width ({width}) should be divisible by 64 for optimal results")
    if steps <= 0:
        raise ValueError("Steps must be positive")
    if guidance < 0:
        raise ValueError("Guidance must be non-negative")


def main():
    """Entry point for Hunyuan-DiT CLI."""
    parser = argparse.ArgumentParser(
        description="Generate images using Hunyuan-DiT model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Required arguments
    parser.add_argument(
        "--prompt",
        type=str,
        required=True,
        help="Text prompt describing the image to generate",
    )

    # Optional arguments
    parser.add_argument(
        "--output",
        type=str,
        default="output.png",
        help="Output file path for generated image",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=50,
        help="Number of inference steps (default: 50 for Hunyuan)",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=1024,
        help="Image height in pixels",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=1024,
        help="Image width in pixels",
    )
    parser.add_argument(
        "--guidance",
        type=float,
        default=7.5,
        help="Guidance scale (default: 7.5 for Hunyuan)",
    )
    parser.add_argument(
        "--quantize",
        type=int,
        choices=[4, 8],
        default=None,
        help="Quantization bit width (4 or 8) for reduced memory usage",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="Path to model weights (local or HuggingFace repo ID)",
    )
    parser.add_argument(
        "--negative-prompt",
        type=str,
        default=None,
        help="Negative prompt for classifier-free guidance",
    )
    parser.add_argument(
        "--lora-paths",
        type=str,
        nargs="*",
        default=None,
        help="Paths to LoRA weights (local files or HuggingFace repos)",
    )
    parser.add_argument(
        "--lora-scales",
        type=float,
        nargs="*",
        default=None,
        help="Scale factors for LoRA weights (default: 1.0 for each)",
    )

    args = parser.parse_args()

    # Validate inputs early
    validate_inputs(args.prompt, args.height, args.width, args.steps, args.guidance)

    try:
        print(f"Loading Hunyuan-DiT model...")
        model = Hunyuan(
            quantize=args.quantize,
            model_path=args.model_path,
            lora_paths=args.lora_paths,
            lora_scales=args.lora_scales,
        )

        print(f"Generating image with prompt: {args.prompt}")
        image = model.generate_image(
            seed=args.seed,
            prompt=args.prompt,
            num_inference_steps=args.steps,
            height=args.height,
            width=args.width,
            guidance=args.guidance,
            negative_prompt=args.negative_prompt,
        )

        image.image.save(args.output)
        print(f"Image saved to: {args.output}")

    except KeyboardInterrupt:
        print("\nGeneration cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
