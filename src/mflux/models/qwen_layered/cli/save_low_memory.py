"""
Memory-efficient save script for Qwen-Image-Layered model.
Loads and saves each component individually to avoid OOM.
"""

import argparse
import gc
import shutil
from pathlib import Path

import mlx.core as mx


def main():
    parser = argparse.ArgumentParser(description="Save quantized Qwen-Image-Layered model (low memory)")
    parser.add_argument("--path", required=True, help="Output directory for saved model")
    parser.add_argument("-q", "--quantize", type=int, choices=[4, 6, 8], required=True, help="Quantization bits")
    parser.add_argument("--model-path", required=True, help="Path to local model weights")
    args = parser.parse_args()

    source_path = Path(args.model_path)
    output_path = Path(args.path)
    output_path.mkdir(parents=True, exist_ok=True)

    print("Copying tokenizer...")
    tokenizer_src = source_path / "tokenizer"
    tokenizer_dst = output_path / "tokenizer"
    if tokenizer_src.exists():
        if tokenizer_dst.exists():
            shutil.rmtree(tokenizer_dst)
        shutil.copytree(tokenizer_src, tokenizer_dst)
        print("  Tokenizer copied")

    from mflux.models.common.weights.loading.weight_applier import WeightApplier
    from mflux.models.common.weights.loading.weight_loader import WeightLoader
    from mflux.models.common.weights.saving.model_saver import ModelSaver
    from mflux.models.qwen_layered.weights.qwen_layered_weight_definition import QwenLayeredWeightDefinition

    component_info = [
        (
            "vae",
            "mflux.models.qwen_layered.model.qwen_layered_vae.qwen_layered_vae",
            "QwenLayeredVAE",
            {"input_channels": 4, "output_channels": 4},
            True,
        ),
        (
            "text_encoder",
            "mflux.models.qwen.model.qwen_text_encoder.qwen_text_encoder",
            "QwenTextEncoder",
            {},
            False,
        ),  # Don't quantize
        ("transformer", "mflux.models.qwen.model.qwen_transformer.qwen_transformer", "QwenTransformer", {}, True),
    ]

    for name, module_path, class_name, kwargs, should_quantize in component_info:
        print(f"\nProcessing {name}...")
        gc.collect()
        mx.metal.clear_cache() if hasattr(mx, "metal") else mx.clear_cache()

        import importlib

        module = importlib.import_module(module_path)
        ComponentClass = getattr(module, class_name)

        # Create component
        component = ComponentClass(**kwargs)

        # Load weights for this component only
        print("  Loading weights...")
        weights = WeightLoader.load(
            weight_definition=QwenLayeredWeightDefinition,
            model_path=str(source_path),
        )

        quant = args.quantize if should_quantize else None
        WeightApplier.apply_and_quantize(
            weights=weights,
            quantize_arg=quant,
            weight_definition=QwenLayeredWeightDefinition,
            models={name: component},
        )
        mx.eval(component.parameters())
        print(f"  Saving to {output_path / name}...")
        ModelSaver._save_weights(str(output_path), args.quantize, component, name)

        del component, weights
        gc.collect()
        mx.metal.clear_cache() if hasattr(mx, "metal") else mx.clear_cache()
        print(f"  {name} done")

    print(f"\nDone! Model saved to {output_path}")
    import subprocess

    result = subprocess.run(["du", "-sh", str(output_path)], capture_output=True, text=True)
    print(f"Size: {result.stdout.strip()}")


if __name__ == "__main__":
    main()
