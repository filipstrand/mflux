"""CLI for NewBie-image generation.

Usage:
    mflux-generate-newbie --prompt "anime girl with blue hair" --steps 28 --guidance 5.0

NewBie-image is optimized for anime/illustration generation with:
- NextDiT transformer with GQA attention
- Dual text encoders (Gemma3 + Jina CLIP)
- 16-channel VAE for high-quality output
"""

import argparse
import os
import sys
import time


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
    """Main CLI entry point for NewBie-image generation."""
    parser = argparse.ArgumentParser(
        description="Generate images using NewBie-image model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Basic generation:
    mflux-generate-newbie --prompt "anime girl with blue hair" --output girl.png

  High quality with more steps:
    mflux-generate-newbie --prompt "detailed fantasy landscape" --steps 50 --guidance 7.0

  With LoRA:
    mflux-generate-newbie --prompt "anime character" --lora-paths ./my_lora --lora-scales 0.8

  Quantized for lower memory:
    mflux-generate-newbie --prompt "test" --quantize 8
        """,
    )

    # Required arguments
    parser.add_argument(
        "--prompt",
        type=str,
        required=True,
        help="Text prompt describing the desired image",
    )

    # Optional generation parameters
    parser.add_argument(
        "--output",
        type=str,
        default="output.png",
        help="Output file path (default: output.png)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility (default: random)",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=28,
        help="Number of inference steps (default: 28)",
    )
    parser.add_argument(
        "--guidance",
        type=float,
        default=5.0,
        help="Classifier-free guidance scale (default: 5.0)",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=1024,
        help="Output image height in pixels (default: 1024)",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=1024,
        help="Output image width in pixels (default: 1024)",
    )

    # Model options
    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="Path to model weights (local or HuggingFace repo ID)",
    )
    parser.add_argument(
        "--quantize",
        type=int,
        choices=[4, 8],
        default=None,
        help="Quantization bit width (4 or 8) for lower memory usage",
    )

    # LoRA options
    parser.add_argument(
        "--lora-paths",
        type=str,
        nargs="+",
        default=None,
        help="Paths to LoRA weights (local files or HuggingFace repos)",
    )
    parser.add_argument(
        "--lora-scales",
        type=float,
        nargs="+",
        default=None,
        help="LoRA scale factors (default: 1.0 for each)",
    )

    # Init image (img2img)
    parser.add_argument(
        "--init-image",
        type=str,
        default=None,
        help="Path to initial image for img2img generation",
    )
    parser.add_argument(
        "--init-strength",
        type=float,
        default=0.3,
        help="Initial image strength (0.0-1.0, default: 0.3)",
    )

    args = parser.parse_args()

    # Generate random seed if not provided
    if args.seed is None:
        import random
        args.seed = random.randint(0, 2**32 - 1)
        print(f"Using random seed: {args.seed}")

    # Validate inputs early
    validate_inputs(args.prompt, args.height, args.width, args.steps, args.guidance)

    # Import model (do this after parsing to avoid slow startup for --help)
    print("Loading NewBie-image model...")
    start_time = time.time()

    from mflux.models.common.config import ModelConfig
    from mflux.models.newbie.variants.txt2img.newbie import NewBie

    model = NewBie(
        model_config=ModelConfig.newbie(),
        quantize=args.quantize,
        model_path=args.model_path,
        lora_paths=args.lora_paths,
        lora_scales=args.lora_scales,
    )

    load_time = time.time() - start_time
    print(f"Model loaded in {load_time:.2f}s")

    # Generate image
    print(f"Generating image with prompt: {args.prompt}")
    print(f"  Steps: {args.steps}, Guidance: {args.guidance}")
    print(f"  Size: {args.width}x{args.height}")

    start_time = time.time()

    image = model.generate_image(
        seed=args.seed,
        prompt=args.prompt,
        num_inference_steps=args.steps,
        height=args.height,
        width=args.width,
        guidance=args.guidance,
        init_image_path=args.init_image,
        init_image_strength=args.init_strength,
    )

    gen_time = time.time() - start_time
    print(f"Generation completed in {gen_time:.2f}s")

    # Save output
    output_path = args.output
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
    image.save(output_path)
    print(f"Image saved to: {output_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
