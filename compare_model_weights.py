#!/usr/bin/env python3
"""
Compare weight structures between Qwen-Image-Edit and Qwen-Image-Edit-2509
to ensure they have the same architecture.
"""

import os

os.environ["HF_HOME"] = os.path.expanduser("~/.cache/huggingface")

from collections import OrderedDict

import torch
from diffusers import DiffusionPipeline


def get_model_state_dict(model_name: str) -> dict:
    """Load model and extract all weight names and shapes."""
    print(f"\n{'=' * 80}")
    print(f"Loading: {model_name}")
    print(f"{'=' * 80}")

    pipe = DiffusionPipeline.from_pretrained(model_name, torch_dtype=torch.bfloat16)

    # Collect all weights from all components
    all_weights = OrderedDict()

    components = {
        "transformer": pipe.transformer,
        "vae": pipe.vae,
        "text_encoder": pipe.text_encoder,
    }

    for component_name, component in components.items():
        if component is None:
            continue
        print(f"\n📦 {component_name.upper()}")
        state_dict = component.state_dict()
        print(f"   Total parameters: {len(state_dict)}")

        for key, tensor in state_dict.items():
            full_key = f"{component_name}.{key}"
            all_weights[full_key] = tensor.shape

    print(f"\n✅ Total weights across all components: {len(all_weights)}")
    return all_weights


def compare_weights(weights1: dict, weights2: dict, name1: str, name2: str):
    """Compare two weight dictionaries."""
    print(f"\n{'=' * 80}")
    print(f"COMPARISON: {name1} vs {name2}")
    print(f"{'=' * 80}")

    keys1 = set(weights1.keys())
    keys2 = set(weights2.keys())

    # Find differences
    only_in_1 = keys1 - keys2
    only_in_2 = keys2 - keys1
    common = keys1 & keys2

    print("\n📊 SUMMARY:")
    print(f"   Weights in {name1}: {len(keys1)}")
    print(f"   Weights in {name2}: {len(keys2)}")
    print(f"   Common weights: {len(common)}")
    print(f"   Only in {name1}: {len(only_in_1)}")
    print(f"   Only in {name2}: {len(only_in_2)}")

    # Check shape differences in common weights
    shape_diffs = [(key, weights1[key], weights2[key]) for key in common if weights1[key] != weights2[key]]

    print(f"   Shape differences: {len(shape_diffs)}")

    # Print details
    if only_in_1:
        print(f"\n❌ ONLY IN {name1}:")
        for key in sorted(only_in_1)[:20]:  # Show first 20
            print(f"   - {key}: {weights1[key]}")
        if len(only_in_1) > 20:
            print(f"   ... and {len(only_in_1) - 20} more")

    if only_in_2:
        print(f"\n❌ ONLY IN {name2}:")
        for key in sorted(only_in_2)[:20]:  # Show first 20
            print(f"   - {key}: {weights2[key]}")
        if len(only_in_2) > 20:
            print(f"   ... and {len(only_in_2) - 20} more")

    if shape_diffs:
        print("\n⚠️  SHAPE DIFFERENCES:")
        for key, shape1, shape2 in shape_diffs[:20]:  # Show first 20
            print(f"   - {key}:")
            print(f"     {name1}: {shape1}")
            print(f"     {name2}: {shape2}")
        if len(shape_diffs) > 20:
            print(f"   ... and {len(shape_diffs) - 20} more")

    # Final verdict
    print(f"\n{'=' * 80}")
    if not only_in_1 and not only_in_2 and not shape_diffs:
        print("✅ IDENTICAL WEIGHT STRUCTURES!")
    else:
        print("❌ WEIGHT STRUCTURES DIFFER!")
    print(f"{'=' * 80}")


def main():
    old_model = "Qwen/Qwen-Image-Edit"
    new_model = "Qwen/Qwen-Image-Edit-2509"

    print("🔍 Comparing Qwen Image Edit model weight structures...\n")

    try:
        # Load both models
        old_weights = get_model_state_dict(old_model)
        new_weights = get_model_state_dict(new_model)

        # Compare
        compare_weights(old_weights, new_weights, "OLD (Qwen-Image-Edit)", "NEW (Qwen-Image-Edit-2509)")

    except (OSError, RuntimeError) as e:
        print(f"\n❌ Error: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
