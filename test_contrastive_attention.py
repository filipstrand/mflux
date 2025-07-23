#!/usr/bin/env python3
"""
Test script for contrastive concept attention.

This script demonstrates the new contrastive attention functionality that uses
"background" as an anti-concept to generate sharper attention heatmaps.

Usage:
    python test_contrastive_attention.py --prompt "a dragon breathing fire" --concept "dragon"
"""

import argparse
from pathlib import Path

from mflux.community.concept_attention.flux_concept import Flux1Concept
from mflux.config.config import Config
from mflux.config.model_config import ModelConfig


def test_contrastive_attention():
    """Test the contrastive concept attention functionality."""

    parser = argparse.ArgumentParser(description="Test contrastive concept attention")
    parser.add_argument("--prompt", type=str, required=True, help="Main generation prompt")
    parser.add_argument("--concept", type=str, required=True, help="Concept to analyze")
    parser.add_argument("--anti-concept", type=str, default="background", help="Anti-concept for contrastive attention")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--height", type=int, default=512, help="Image height")
    parser.add_argument("--width", type=int, default=512, help="Image width")
    parser.add_argument("--steps", type=int, default=20, help="Number of inference steps")
    parser.add_argument("--output", type=str, default="./output", help="Output directory")
    parser.add_argument("--model", type=str, default="dev", help="Model variant (dev/schnell)")
    parser.add_argument("--sharpening", type=float, default=2.0, help="Spectral sharpening exponent")
    parser.add_argument("--temperature", type=float, default=0.1, help="Softmax temperature")

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(exist_ok=True)

    print("Testing contrastive concept attention:")
    print(f"  Prompt: {args.prompt}")
    print(f"  Concept: {args.concept}")
    print(f"  Anti-concept: {args.anti_concept}")
    print(f"  Sharpening exponent: {args.sharpening}")
    print(f"  Temperature: {args.temperature}")
    print()

    # Initialize model
    model_config = ModelConfig.dev() if args.model == "dev" else ModelConfig.schnell()
    flux_model = Flux1Concept(model_config=model_config)

    # Create config
    config = Config(
        num_inference_steps=args.steps,
        height=args.height,
        width=args.width,
        guidance=3.5,
    )

    print("Generating contrastive concept attention heatmap...")

    # Generate contrastive concept attention
    contrastive_result = flux_model.generate_contrastive_image(
        seed=args.seed,
        prompt=args.prompt,
        concept=args.concept,
        config=config,
        anti_concept=args.anti_concept,
        sharpening_exponent=args.sharpening,
        temperature=args.temperature,
    )

    # Save contrastive result
    contrastive_path = output_dir / f"contrastive_{args.concept.replace(' ', '_')}_{args.seed}.png"
    contrastive_result.image.save(contrastive_path)
    contrastive_result.concept_heatmap.save(
        output_dir / f"contrastive_heatmap_{args.concept.replace(' ', '_')}_{args.seed}.png"
    )

    print(f"Contrastive result saved to: {contrastive_path}")

    print("\nContrastive heatmap generated:")
    contrastive_heatmap_name = f"contrastive_heatmap_{args.concept.replace(' ', '_')}_{args.seed}.png"
    print(f"  Contrastive heatmap: {output_dir / contrastive_heatmap_name}")
    print()
    print("The contrastive heatmap should show:")
    print("  - Sharper boundaries around the concept")
    print("  - Less 'leakage' into surrounding areas")
    print(f"  - More defined contrast between '{args.concept}' and '{args.anti_concept}'")


def test_parameter_sensitivity():
    """Test how different parameters affect the sharpening."""

    print("Testing parameter sensitivity...")

    # Test different sharpening exponents
    sharpening_values = [1.0, 1.5, 2.0, 3.0, 5.0]
    temperature_values = [1.0, 0.5, 0.1, 0.05, 0.01]

    model_config = ModelConfig.schnell()  # Use faster model for testing
    flux_model = Flux1Concept(model_config=model_config)

    config = Config(
        num_inference_steps=10,  # Fewer steps for faster testing
        height=256,
        width=256,
        guidance=3.5,
    )

    prompt = "a red dragon on a mountain"
    concept = "dragon"
    seed = 123

    output_dir = Path("./parameter_test")
    output_dir.mkdir(exist_ok=True)

    # Test sharpening exponents
    print("Testing sharpening exponents...")
    for sharpening in sharpening_values:
        print(f"  Testing sharpening={sharpening}")
        result = flux_model.generate_contrastive_image(
            seed=seed,
            prompt=prompt,
            concept=concept,
            config=config,
            anti_concept="background",
            sharpening_exponent=sharpening,
            temperature=0.1,
        )
        result.concept_heatmap.save(output_dir / f"sharp_{sharpening:.1f}_temp_0.1.png")

    # Test temperature values
    print("Testing temperature values...")
    for temp in temperature_values:
        print(f"  Testing temperature={temp}")
        result = flux_model.generate_contrastive_image(
            seed=seed,
            prompt=prompt,
            concept=concept,
            config=config,
            anti_concept="background",
            sharpening_exponent=2.0,
            temperature=temp,
        )
        result.concept_heatmap.save(output_dir / f"sharp_2.0_temp_{temp:.2f}.png")

    print(f"Parameter test results saved to: {output_dir}")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "--test-params":
        test_parameter_sensitivity()
    else:
        test_contrastive_attention()
