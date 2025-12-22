#!/usr/bin/env python
"""CLI for Qwen-Image-Layered: Image decomposition into RGBA layers."""

import argparse
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description="Decompose an image into RGBA layers using Qwen-Image-Layered.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic decomposition into 4 layers
  mflux-generate-qwen-layered --image input.png --layers 4

  # With 6-bit quantization (recommended for 48GB Macs)
  mflux-generate-qwen-layered --image input.png --layers 4 -q 6

  # Custom settings
  mflux-generate-qwen-layered --image input.png --layers 8 --steps 50 --resolution 640 -q 6
        """,
    )

    # Required arguments
    parser.add_argument(
        "--image",
        type=Path,
        required=True,
        help="Path to input image to decompose",
    )

    # Optional arguments
    parser.add_argument(
        "--layers",
        type=int,
        default=4,
        help="Number of output layers (default: 4)",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=50,
        help="Number of inference steps (default: 50)",
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=640,
        choices=[640, 1024],
        help="Target resolution bucket (default: 640)",
    )
    parser.add_argument(
        "--guidance",
        type=float,
        default=4.0,
        help="Guidance scale (default: 4.0)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="Optional text prompt describing the image",
    )
    parser.add_argument(
        "--negative-prompt",
        type=str,
        default=" ",
        help="Optional negative prompt",
    )
    parser.add_argument(
        "-q",
        "--quantize",
        type=int,
        choices=[4, 6, 8],
        default=None,
        help="Quantization bits (4, 6, or 8). Recommended: 6 for 48GB Macs",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="Path to local model weights (otherwise downloads from HuggingFace)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="layer_{i}.png",
        help="Output filename pattern. Use {i} for layer index (default: layer_{i}.png)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("."),
        help="Output directory (default: current directory)",
    )

    args = parser.parse_args()

    # Validate input
    if not args.image.exists():
        print(f"Error: Input image not found: {args.image}")
        sys.exit(1)

    # Import here to avoid slow startup for --help
    from mflux.models.qwen_layered.variants.i2l.qwen_image_layered import QwenImageLayered

    print("=" * 60)
    print("Qwen-Image-Layered: Image Decomposition")
    print("=" * 60)
    print(f"  Input: {args.image}")
    print(f"  Layers: {args.layers}")
    print(f"  Steps: {args.steps}")
    print(f"  Resolution: {args.resolution}")
    print(f"  Guidance: {args.guidance}")
    print(f"  Seed: {args.seed}")
    print(f"  Quantization: {args.quantize or 'BF16 (full precision)'}")
    print("=" * 60)

    # Initialize model
    print("\nLoading model...")
    model = QwenImageLayered(
        quantize=args.quantize,
        model_path=args.model_path,
    )

    # Run decomposition
    print("\nStarting decomposition...")
    layers = model.decompose(
        seed=args.seed,
        image_path=args.image,
        num_layers=args.layers,
        num_inference_steps=args.steps,
        guidance=args.guidance,
        resolution=args.resolution,
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
    )

    # Save output layers
    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nSaving {len(layers)} layers to {args.output_dir}/")

    for i, layer in enumerate(layers):
        filename = args.output.format(i=i)
        output_path = args.output_dir / filename
        layer.save(output_path)
        print(f"  Saved {output_path}")

    print("\nDone!")


if __name__ == "__main__":
    main()
