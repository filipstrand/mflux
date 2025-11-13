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
from pathlib import Path

import mlx.core as mx

from mflux.config.model_config import ModelConfig
from mflux_debugger.weight_inspector import WeightInspector


def load_weights_generic(model_name: str, local_path: str | None = None) -> tuple[dict, dict]:
    """
    Generic weight loader - just loads weights from HuggingFace or local path.

    Simple approach:
    1. Try HuggingFace safetensors files (mx.load)
    2. Fallback to PyTorch model loading (transformers/diffusers)

    Returns:
        Tuple of (raw_weights, mapped_weights)
        For generic loading, mapped_weights = raw_weights (no mapping applied)
    """
    # Try HuggingFace safetensors first (simplest, most generic)
    print("   Loading weights from HuggingFace...")
    try:
        from huggingface_hub import snapshot_download

        # Download or get cached model
        if local_path:
            model_path = Path(local_path)
        else:
            # Try cached first
            try:
                cache_dir = snapshot_download(repo_id=model_name, local_files_only=True)
                model_path = Path(cache_dir)
            except (FileNotFoundError, ValueError, OSError):
                # Download if not cached
                print(f"   Downloading {model_name} from HuggingFace...")
                model_path = Path(snapshot_download(repo_id=model_name))

        # Load all safetensors files
        raw_weights = {}
        safetensors_files = list(model_path.glob("**/*.safetensors"))

        if safetensors_files:
            print(f"   Found {len(safetensors_files)} safetensors files")
            # Load all files, collect errors
            # Note: try-except in loop is intentional - we want to continue loading
            # other files even if one fails
            load_errors = []
            for safetensor_file in safetensors_files:
                try:
                    data = mx.load(str(safetensor_file), return_metadata=True)
                    raw_weights.update(dict(data[0].items()))
                except (OSError, ValueError, RuntimeError) as e:  # noqa: PERF203
                    load_errors.append(f"{safetensor_file.name}: {e}")

            if load_errors:
                print(f"   Warning: Could not load {len(load_errors)} files: {load_errors[0]}")

            if raw_weights:
                # For generic loading, mapped_weights = raw_weights (no mapping)
                print(f"   Loaded {len(raw_weights)} weights")
                return raw_weights, raw_weights.copy()
        else:
            print(f"   No safetensors files found in {model_path}")
    except (FileNotFoundError, ValueError, OSError, RuntimeError) as e:
        print(f"   HuggingFace loading failed: {e}")

    # Fallback: Try PyTorch model loading
    print("   Attempting PyTorch model loading...")
    try:
        import torch

        # Try diffusers first (for diffusion models)
        try:
            from diffusers import DiffusionPipeline

            pipe = DiffusionPipeline.from_pretrained(
                model_name if not local_path else local_path, torch_dtype=torch.bfloat16
            )

            # Extract weights from all pipeline components
            raw_weights = {}
            for component_name in ["vae", "text_encoder", "unet", "transformer", "decoder", "encoder"]:
                component = getattr(pipe, component_name, None)
                if component is not None:
                    try:
                        state_dict = component.state_dict()
                        for key, tensor in state_dict.items():
                            if tensor.dtype == torch.bfloat16:
                                tensor = tensor.to(torch.float32)
                            raw_weights[f"{component_name}.{key}"] = mx.array(tensor.detach().cpu().numpy())
                    except (AttributeError, RuntimeError, OSError) as e:
                        print(f"   Warning: Could not extract {component_name}: {e}")

            if raw_weights:
                print(f"   Loaded {len(raw_weights)} weights from PyTorch pipeline")
                return raw_weights, raw_weights.copy()
        except (ImportError, RuntimeError, OSError, ValueError) as e:
            print(f"   Diffusers loading failed: {e}")

        # Try transformers
        try:
            from transformers import AutoModel

            model = AutoModel.from_pretrained(model_name if not local_path else local_path)
            state_dict = model.state_dict()

            raw_weights = {}
            for key, tensor in state_dict.items():
                if tensor.dtype == torch.bfloat16:
                    tensor = tensor.to(torch.float32)
                raw_weights[key] = mx.array(tensor.detach().cpu().numpy())

            if raw_weights:
                print(f"   Loaded {len(raw_weights)} weights from transformers model")
                return raw_weights, raw_weights.copy()
        except (ImportError, RuntimeError, OSError, ValueError) as e:
            print(f"   Transformers loading failed: {e}")
    except (ImportError, RuntimeError, OSError) as e:
        print(f"   PyTorch loading failed: {e}")

    raise ValueError(
        f"Could not load weights for {model_name}. "
        "Tried: HuggingFace safetensors, PyTorch models (diffusers/transformers). "
        "Please ensure the model is available or provide --local-path."
    )


def cmd_tutorial(lesson: str = "basic") -> int:
    """Show interactive tutorial for the weight inspector."""
    if lesson == "basic":
        _show_basic_tutorial()
        return 0
    else:
        print(f"❌ Unknown tutorial: {lesson}", file=sys.stderr)
        return 1


def _show_basic_tutorial():
    """Show step-by-step guide for basic weight inspection tutorial."""
    print("🎓 Interactive Weight Inspector Tutorial")
    print("=" * 70)
    print()
    print("Learn how to inspect and analyze model weights interactively.")
    print("This tutorial teaches you the essential commands step-by-step.")
    print()
    print("📚 What You'll Learn:")
    print("  • How to load weights from any HuggingFace model")
    print("  • How to explore weight structure and components")
    print("  • How to search for specific weights")
    print("  • How to inspect individual weight tensors")
    print("  • How to generate comprehensive verification reports")
    print()
    print("=" * 70)
    print()
    print("🎯 Step 1: Basic Weight Inspection")
    print("-" * 70)
    print("Start by inspecting a model's weights. The tool loads safetensors")
    print("directly from HuggingFace (or falls back to PyTorch).")
    print()
    print("Command:")
    print("  mflux-debug-inspect-weights briaai/FIBO")
    print()
    print("What happens:")
    print("  • Downloads/loads weights from HuggingFace")
    print("  • Shows summary: total weights, components, counts")
    print("  • Shows structure: organized by component (flat or nested)")
    print()
    print("💡 Try it now! Press Enter to continue...")
    input()

    print()
    print("🎯 Step 2: Filter by Component")
    print("-" * 70)
    print("Focus on a specific component (e.g., decoder, transformer_blocks).")
    print()
    print("Command:")
    print("  mflux-debug-inspect-weights briaai/FIBO --component decoder")
    print()
    print("What happens:")
    print("  • Shows only weights matching the component prefix")
    print("  • Organized by component with first few weights displayed")
    print("  • Useful for exploring large models")
    print()
    print("💡 Try it now! Press Enter to continue...")
    input()

    print()
    print("🎯 Step 3: Search for Weights")
    print("-" * 70)
    print("Find weights matching a pattern (e.g., 'conv', 'norm', 'attn').")
    print()
    print("Command:")
    print("  mflux-debug-inspect-weights briaai/FIBO --search conv_in")
    print()
    print("What happens:")
    print("  • Lists all weights matching the search pattern")
    print("  • Shows up to 50 results (truncated if more)")
    print("  • Useful for finding specific weight names")
    print()
    print("💡 Try it now! Press Enter to continue...")
    input()

    print()
    print("🎯 Step 4: Inspect Specific Weight")
    print("-" * 70)
    print("Get detailed information about a single weight tensor.")
    print()
    print("Command:")
    print("  mflux-debug-inspect-weights briaai/FIBO --weight decoder.conv_in.weight")
    print()
    print("What happens:")
    print("  • Shows full tensor statistics (shape, dtype, size)")
    print("  • Shows min/max/mean/std values")
    print("  • Useful for debugging weight loading/mapping")
    print()
    print("💡 Try it now! Press Enter to continue...")
    input()

    print()
    print("🎯 Step 5: Full Verification Report")
    print("-" * 70)
    print("Generate a comprehensive report with pattern detection,")
    print("coverage analysis, and structure verification.")
    print()
    print("Command:")
    print("  mflux-debug-inspect-weights briaai/FIBO --report")
    print()
    print("What happens:")
    print("  • Detects patterns (blocks, layers, components)")
    print("  • Analyzes mapping coverage (if mapped weights exist)")
    print("  • Verifies structure correctness")
    print("  • Shows actual nested/flat structure")
    print("  • Useful for verifying weight mappings")
    print()
    print("💡 Try it now! Press Enter to continue...")
    input()

    print()
    print("🎯 Step 6: Using Local Paths")
    print("-" * 70)
    print("Load weights from a local directory instead of HuggingFace.")
    print()
    print("Command:")
    print("  mflux-debug-inspect-weights model-name --local-path /path/to/model")
    print()
    print("What happens:")
    print("  • Loads safetensors from local directory")
    print("  • Falls back to PyTorch if safetensors not found")
    print("  • Useful for offline work or custom models")
    print()
    print("💡 Try it now! Press Enter to continue...")
    input()

    print()
    print("=" * 70)
    print("✅ Tutorial Complete!")
    print("=" * 70)
    print()
    print("📖 Key Takeaways:")
    print("  • Default mode: Summary + Structure (most useful)")
    print("  • --component: Filter by component prefix")
    print("  • --search: Find weights by pattern")
    print("  • --weight: Inspect specific weight tensor")
    print("  • --report: Full verification report")
    print("  • --local-path: Load from local directory")
    print()
    print("💡 Tips:")
    print("  • Works with any HuggingFace model (no handlers needed)")
    print("  • Handles both flat (raw) and nested (mapped) structures")
    print("  • Shows actual data - never lies about structure")
    print("  • Use --help anytime for quick reference")
    print()
    print("🚀 Next Steps:")
    print("  • Try inspecting different models (Qwen, Flux, FIBO, etc.)")
    print("  • Use --report to verify weight mappings")
    print("  • Use --search to explore model architecture")
    print()


def load_weights(model_config_or_name: ModelConfig | str, local_path: str | None = None) -> tuple[dict, dict]:
    """
    Load weights for the given model.

    Simple, generic approach:
    - Model name is just a location identifier (HuggingFace repo)
    - Loads safetensors files directly using mx.load()
    - Falls back to PyTorch if needed
    - No model-specific handlers needed

    Works with:
    - ModelConfig objects (extracts model_name)
    - Model names as strings (e.g., "briaai/FIBO", "Qwen/Qwen-Image")

    Returns:
        Tuple of (raw_weights, mapped_weights)
        For generic loading, mapped_weights = raw_weights (no mapping applied)
    """
    # Extract model name if ModelConfig provided
    if isinstance(model_config_or_name, ModelConfig):
        model_name = model_config_or_name.model_name
    else:
        model_name = model_config_or_name

    # Always use generic loading - simple and works for any model
    return load_weights_generic(model_name, local_path)


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

  # Full verification report (patterns, coverage, structure)
  mflux-debug-inspect-weights qwen-image --report

  # Inspect specific weight
  mflux-debug-inspect-weights qwen-image --weight "transformer_blocks.0.attn.to_q.weight"

  # Filter by component (shows structure for that component only)
  mflux-debug-inspect-weights qwen-image --component decoder

  # Interactive tutorial
  mflux-debug-inspect-weights tutorial
        """,
    )

    parser.add_argument(
        "model_name", nargs="?", help="Model name (e.g., 'Qwen/Qwen-Image' or 'qwen-image') or 'tutorial'"
    )
    parser.add_argument("--local-path", type=str, help="Local path to model weights")
    parser.add_argument("--component", type=str, help="Filter by component (e.g., 'transformer_blocks', 'decoder')")
    parser.add_argument("--search", type=str, help="Search pattern for weights")
    parser.add_argument("--weight", type=str, help="Inspect specific weight path")
    parser.add_argument(
        "--report", action="store_true", help="Print full verification report (patterns, coverage, structure)"
    )
    parser.add_argument("--format", choices=["hf", "mlx"], default="hf", help="Weight format (default: hf)")

    args = parser.parse_args()

    # Handle tutorial command
    if args.model_name == "tutorial":
        return cmd_tutorial("basic")

    if args.model_name is None:
        parser.print_help()
        return 1

    try:
        # Try to resolve as ModelConfig first (for known models)
        model_name = args.model_name
        try:
            print(f"🔍 Resolving model: {model_name}")
            model_config = ModelConfig.from_name(model_name)
            print(f"✅ Model: {model_config.model_name}")
            # Use ModelConfig for loading
            model_name_for_loading = model_config
        except (ValueError, KeyError, AttributeError):
            # Not in ModelConfig - use as generic model name
            print(f"🔍 Model not in ModelConfig, using generic loader: {model_name}")
            model_name_for_loading = model_name

        # Load weights (works with both ModelConfig and string)
        print("\n📦 Loading weights...")
        raw_weights, mapped_weights = load_weights(model_name_for_loading, args.local_path)
        print("✅ Loaded weights")

        # Create inspector (use actual model name string)
        if isinstance(model_name_for_loading, ModelConfig):
            display_name = model_name_for_loading.model_name
        else:
            display_name = model_name
        inspector = WeightInspector(raw_weights, mapped_weights, display_name)

        # Execute requested action
        if args.report:
            # Full verification report (always includes structure)
            inspector.print_mapping_report()
        elif args.weight:
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
        else:
            # Default: Summary + Structure (always show structure - most useful)
            inspector.print_summary()
            print()  # Blank line
            inspector.print_structure(max_depth=4, show_types=True, component=args.component)

    except (ValueError, FileNotFoundError, KeyError, AttributeError) as e:
        print(f"❌ Error: {e}", file=sys.stderr)
        import traceback

        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    main()
