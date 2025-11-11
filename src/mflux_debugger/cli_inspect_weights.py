#!/usr/bin/env python3
"""
CLI command for inspecting model weights.

Usage:
    mflux-debug-inspect-weights <model_name> [options]

Examples:
    mflux-debug-inspect-weights Qwen/Qwen-Image
    mflux-debug-inspect-weights qwen-image-edit --component transformer_blocks
    mflux-debug-inspect-weights black-forest-labs/FLUX.1-dev --search "attn.to_q"
"""

import argparse
import sys

from mflux.config.model_config import ModelConfig
from mflux.models.flux.weights.weight_handler import WeightHandler as FluxWeightHandler
from mflux.models.qwen.weights.qwen_weight_handler import QwenWeightHandler
from mflux_debugger.weight_inspector import WeightInspector


def detect_model_type(model_config: ModelConfig) -> str:
    """Detect if model is Qwen or Flux based on model name."""
    model_name_lower = model_config.model_name.lower()
    if "qwen" in model_name_lower:
        return "qwen"
    elif "flux" in model_name_lower:
        return "flux"
    else:
        # Try to infer from aliases
        for alias in model_config.aliases:
            if "qwen" in alias.lower():
                return "qwen"
            elif "flux" in alias.lower():
                return "flux"
        return "unknown"


def load_weights(model_config: ModelConfig, local_path: str | None = None) -> tuple[dict, dict]:
    """
    Load weights for the given model.

    Returns:
        Tuple of (raw_weights, mapped_weights)
    """
    model_type = detect_model_type(model_config)

    if model_type == "qwen":
        # First, get the root path
        from pathlib import Path

        from mflux.models.flux.weights.weight_handler import WeightHandler

        root_path = (
            Path(local_path) if local_path else WeightHandler.download_or_get_cached_weights(model_config.model_name)
        )

        # Load raw weights before mapping
        print("   Loading raw HuggingFace weights...")
        raw_transformer = QwenWeightHandler._load_safetensors_shards(
            root_path / "transformer", loading_mode="multi_glob"
        )
        raw_text_encoder = QwenWeightHandler._load_safetensors_shards(
            root_path / "text_encoder", loading_mode="multi_json"
        )
        raw_vae = QwenWeightHandler._load_safetensors_shards(root_path / "vae", loading_mode="single")

        # Combine all raw weights
        raw_weights = {}
        raw_weights.update(raw_transformer)
        raw_weights.update(raw_text_encoder)
        raw_weights.update(raw_vae)

        # Load mapped weights using the handler
        print("   Loading mapped MLX weights...")
        weight_handler = QwenWeightHandler.load_regular_weights(
            repo_id=model_config.model_name,
            local_path=local_path,
        )

        # Get mapped weights - flatten nested structure for easier inspection
        def flatten_dict(d: dict, prefix: str = "") -> dict:
            """Flatten nested dict to dot-notation keys."""
            result = {}
            for k, v in d.items():
                key = f"{prefix}.{k}" if prefix else k
                if isinstance(v, dict):
                    result.update(flatten_dict(v, key))
                else:
                    result[key] = v
            return result

        mapped_weights_flat = {}
        if weight_handler.transformer:
            mapped_weights_flat.update(flatten_dict(weight_handler.transformer, "transformer"))
        if weight_handler.qwen_text_encoder:
            mapped_weights_flat.update(flatten_dict(weight_handler.qwen_text_encoder, "qwen_text_encoder"))
        if weight_handler.vae:
            mapped_weights_flat.update(flatten_dict(weight_handler.vae, "vae"))

        return raw_weights, mapped_weights_flat

    elif model_type == "flux":
        # Similar approach for Flux
        from pathlib import Path

        from mflux.models.flux.weights.weight_handler import WeightHandler

        root_path = (
            Path(local_path) if local_path else WeightHandler.download_or_get_cached_weights(model_config.model_name)
        )

        print("   Loading raw HuggingFace weights...")
        # Load raw weights using Flux handler's internal methods
        # Note: Flux handler structure is different, so we'll focus on mapped weights for now
        raw_weights = {}  # TODO: Extract raw weights from Flux handler

        print("   Loading mapped MLX weights...")
        weight_handler = FluxWeightHandler.load_regular_weights(
            repo_id=model_config.model_name,
            local_path=local_path,
            transformer_repo_id=model_config.custom_transformer_model,
        )

        # Flatten mapped weights
        def flatten_dict(d: dict, prefix: str = "") -> dict:
            """Flatten nested dict to dot-notation keys."""
            result = {}
            for k, v in d.items():
                key = f"{prefix}.{k}" if prefix else k
                if isinstance(v, dict):
                    result.update(flatten_dict(v, key))
                else:
                    result[key] = v
            return result

        mapped_weights_flat = {}
        if weight_handler.transformer:
            mapped_weights_flat.update(flatten_dict(weight_handler.transformer, "transformer"))
        if weight_handler.clip_encoder:
            mapped_weights_flat.update(flatten_dict(weight_handler.clip_encoder, "clip_encoder"))
        if weight_handler.t5_encoder:
            mapped_weights_flat.update(flatten_dict(weight_handler.t5_encoder, "t5_encoder"))
        if weight_handler.vae:
            mapped_weights_flat.update(flatten_dict(weight_handler.vae, "vae"))

        return raw_weights, mapped_weights_flat

    else:
        raise ValueError(f"Unknown model type for: {model_config.model_name}")


def main():
    parser = argparse.ArgumentParser(
        prog="mflux-debug-inspect-weights",
        description="Inspect and visualize model weights",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Inspect Qwen Image model
  mflux-debug-inspect-weights Qwen/Qwen-Image

  # Inspect with alias
  mflux-debug-inspect-weights qwen-image-edit

  # Show only transformer blocks
  mflux-debug-inspect-weights qwen-image --component transformer_blocks

  # Search for specific weights
  mflux-debug-inspect-weights qwen-image --search "attn.to_q"

  # Show tree view
  mflux-debug-inspect-weights qwen-image --tree

  # Inspect specific weight
  mflux-debug-inspect-weights qwen-image --weight "transformer_blocks.0.attn.to_q.weight"
        """,
    )

    parser.add_argument("model_name", help="Model name (e.g., 'Qwen/Qwen-Image' or 'qwen-image')")
    parser.add_argument("--local-path", type=str, help="Local path to model weights")
    parser.add_argument("--component", type=str, help="Filter by component (e.g., 'transformer_blocks')")
    parser.add_argument("--search", type=str, help="Search pattern for weights")
    parser.add_argument("--weight", type=str, help="Inspect specific weight path")
    parser.add_argument("--tree", action="store_true", help="Show tree view of weights")
    parser.add_argument("--format", choices=["hf", "mlx"], default="hf", help="Weight format (default: hf)")

    args = parser.parse_args()

    try:
        # Resolve model config
        print(f"🔍 Resolving model: {args.model_name}")
        model_config = ModelConfig.from_name(args.model_name)
        print(f"✅ Model: {model_config.model_name}")

        # Load weights
        print("\n📦 Loading weights...")
        raw_weights, mapped_weights = load_weights(model_config, args.local_path)
        print("✅ Loaded weights")

        # Create inspector (use raw_weights as primary, mapped_weights as secondary)
        # For now, we'll use raw_weights for inspection since they're more complete
        inspector = WeightInspector(raw_weights, {"mapped": mapped_weights}, model_config.model_name)

        # Execute requested action
        if args.weight:
            # Inspect specific weight
            inspector.pretty_print(args.weight, format=args.format)
        elif args.search:
            # Search weights
            matches = inspector.search(args.search, format=args.format)
            print(f"\n🔍 Found {len(matches)} matching weights:")
            for match in matches[:50]:  # Limit to 50 results
                print(f"   {match}")
            if len(matches) > 50:
                print(f"   ... and {len(matches) - 50} more")
        elif args.tree:
            # Tree view
            inspector.print_tree(component=args.component)
        else:
            # Summary view
            inspector.print_summary()
            if args.component:
                print(f"\n📋 Weights in '{args.component}':")
                matches = inspector.search(args.component, format=args.format)
                for match in matches[:20]:  # Show first 20
                    print(f"   {match}")
                if len(matches) > 20:
                    print(f"   ... and {len(matches) - 20} more")

    except (ValueError, FileNotFoundError, KeyError, AttributeError) as e:
        print(f"❌ Error: {e}", file=sys.stderr)
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
