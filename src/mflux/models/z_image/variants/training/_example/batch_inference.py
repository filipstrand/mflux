#!/usr/bin/env python3
"""Batch inference script for Z-Image-Base.

Optimized for Mac Studio M3 Ultra with 512GB unified memory.
Supports batch inference for maximum throughput.

Usage:
    python batch_inference.py --prompts prompts.txt --output-dir ./outputs
    python batch_inference.py --prompts prompts.txt --batch-size 8
"""

import argparse
import time
from pathlib import Path

import mlx.core as mx

from mflux.models.z_image.variants.training.z_image_base import ZImageBase


def load_prompts(prompts_file: Path) -> list[str]:
    """Load prompts from a text file (one per line)."""
    with open(prompts_file, "r") as f:
        prompts = [line.strip() for line in f if line.strip()]
    return prompts


def estimate_memory_usage(batch_size: int, width: int, height: int) -> float:
    """Estimate memory usage in GB."""
    # Base model: ~12GB
    base_model = 12.0

    # Per-image latent: width * height * 16 channels * 4 bytes / 8^2 downscale
    latent_size = (width * height * 16 * 4) / (8 * 8 * 1024 * 1024 * 1024)

    # Embeddings: ~1GB per batch
    embeddings = batch_size * 0.1

    # Intermediate activations: ~8GB per image
    activations = batch_size * 8.0

    return base_model + batch_size * latent_size + embeddings + activations


def main():
    parser = argparse.ArgumentParser(description="Batch inference for Z-Image-Base")
    parser.add_argument("--prompts", type=Path, required=True, help="Path to prompts file (one per line)")
    parser.add_argument("--negative-prompt", type=str, default="", help="Negative prompt for all images")
    parser.add_argument("--output-dir", type=Path, default=Path("./batch_outputs"), help="Output directory")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size (default: 8)")
    parser.add_argument("--width", type=int, default=1024, help="Image width")
    parser.add_argument("--height", type=int, default=1024, help="Image height")
    parser.add_argument("--steps", type=int, default=50, help="Inference steps")
    parser.add_argument("--guidance", type=float, default=3.5, help="CFG guidance scale")
    parser.add_argument("--quantize", type=int, choices=[4, 8], default=None, help="Quantization level")
    parser.add_argument("--seed", type=int, default=42, help="Base seed")
    parser.add_argument("--lora-path", type=str, default=None, help="Path to LoRA weights")
    parser.add_argument("--lora-scale", type=float, default=1.0, help="LoRA scale")
    args = parser.parse_args()

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Load prompts
    prompts = load_prompts(args.prompts)
    print(f"Loaded {len(prompts)} prompts")

    # Estimate memory usage
    estimated_memory = estimate_memory_usage(args.batch_size, args.width, args.height)
    print(f"Estimated memory usage: {estimated_memory:.1f} GB")

    if estimated_memory > 450:
        suggested_batch = max(1, int(args.batch_size * 450 / estimated_memory))
        print(f"Warning: High memory usage. Consider reducing batch_size to {suggested_batch}")

    # Load model
    print("Loading Z-Image-Base model...")
    lora_paths = [args.lora_path] if args.lora_path else None
    lora_scales = [args.lora_scale] if args.lora_path else None

    model = ZImageBase(
        quantize=args.quantize,
        lora_paths=lora_paths,
        lora_scales=lora_scales,
    )
    print("Model loaded")

    # Process in batches
    total_time = 0
    total_images = 0

    for batch_idx in range(0, len(prompts), args.batch_size):
        batch_prompts = prompts[batch_idx : batch_idx + args.batch_size]
        batch_start = time.time()

        for i, prompt in enumerate(batch_prompts):
            image_idx = batch_idx + i
            seed = args.seed + image_idx

            print(f"Generating image {image_idx + 1}/{len(prompts)}: {prompt[:50]}...")

            image = model.generate_image(
                seed=seed,
                prompt=prompt,
                negative_prompt=args.negative_prompt,
                width=args.width,
                height=args.height,
                guidance_scale=args.guidance,
                num_inference_steps=args.steps,
            )

            # Save image
            output_path = args.output_dir / f"image_{image_idx:04d}.png"
            image.save(output_path)

            # Force memory sync
            mx.synchronize()

        batch_time = time.time() - batch_start
        total_time += batch_time
        total_images += len(batch_prompts)

        images_per_second = len(batch_prompts) / batch_time
        print(f"Batch {batch_idx // args.batch_size + 1}: {images_per_second:.2f} images/sec")

    # Summary
    print("\n=== Summary ===")
    print(f"Total images: {total_images}")
    print(f"Total time: {total_time:.1f} seconds")
    print(f"Average time per image: {total_time / total_images:.1f} seconds")
    print(f"Images per second: {total_images / total_time:.2f}")
    print(f"Output directory: {args.output_dir}")


if __name__ == "__main__":
    main()
