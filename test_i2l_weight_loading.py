"""Test weight loading for Z-Image i2L models.

Run: python test_i2l_weight_loading.py

This will:
1. Download weights from HuggingFace (cached after first run)
2. Create MLX model instances
3. Load weights into models
4. Report success/failure and memory usage
"""

import sys
import time

import mlx.core as mx
import mlx.nn as nn

# Add src to path
sys.path.insert(0, "src")

from mflux.models.z_image.model.z_image_i2l.dinov3.dinov3_vision_transformer import DINOv3VisionTransformer
from mflux.models.z_image.model.z_image_i2l.i2l_decoder.i2l_decoder import ZImageI2LDecoder
from mflux.models.z_image.model.z_image_i2l.i2l_weight_loader import (
    load_dinov3_weights,
    load_i2l_decoder_weights,
    load_siglip2_weights,
)
from mflux.models.z_image.model.z_image_i2l.siglip2.siglip2_vision_transformer import Siglip2VisionTransformer


def count_params(model: nn.Module) -> int:
    """Count total parameters in a model."""
    params = model.parameters()
    from mlx.utils import tree_flatten

    return sum(v.size for _, v in tree_flatten(params))


def test_model_loading(name: str, model_cls, weight_loader):
    """Test loading weights into a model."""
    print(f"\n{'=' * 60}")
    print(f"Testing: {name}")
    print(f"{'=' * 60}")

    # Create model
    t0 = time.time()
    model = model_cls()
    mx.eval(model.parameters())
    t1 = time.time()
    print(f"  Model created in {t1 - t0:.2f}s")
    print(f"  Parameters: {count_params(model):,}")

    # Load weights
    t0 = time.time()
    weights = weight_loader()
    t1 = time.time()
    print(f"  Weights downloaded/loaded in {t1 - t0:.2f}s")

    # Apply weights
    t0 = time.time()
    model.update(weights, strict=False)
    mx.eval(model.parameters())
    t1 = time.time()
    print(f"  Weights applied in {t1 - t0:.2f}s")

    return model


def test_forward_pass(name: str, model, input_tensor):
    """Test a forward pass through the model."""
    print(f"\n  Forward pass test ({name}):")
    print(f"    Input shape: {input_tensor.shape}")
    t0 = time.time()
    output = model(input_tensor)
    mx.eval(output)
    t1 = time.time()
    print(f"    Output shape: {output.shape}")
    print(f"    Output dtype: {output.dtype}")
    print(f"    Time: {t1 - t0:.2f}s")
    print(f"    Output range: [{output.min().item():.4f}, {output.max().item():.4f}]")
    return output


if __name__ == "__main__":
    print("Z-Image i2L Weight Loading Test")
    print("=" * 60)

    # 1. Test SigLIP2-G384
    siglip2 = test_model_loading("SigLIP2-G384", Siglip2VisionTransformer, load_siglip2_weights)
    # Forward pass with dummy image
    dummy_img = mx.random.normal(shape=(1, 3, 384, 384)).astype(mx.bfloat16)
    siglip2_out = test_forward_pass("SigLIP2", siglip2, dummy_img)

    # 2. Test DINOv3-7B
    dinov3 = test_model_loading("DINOv3-7B", DINOv3VisionTransformer, load_dinov3_weights)
    # Forward pass with dummy image
    dummy_img = mx.random.normal(shape=(1, 3, 224, 224)).astype(mx.bfloat16)
    dinov3_out = test_forward_pass("DINOv3", dinov3, dummy_img)

    # 3. Test i2L Decoder
    i2l = test_model_loading("i2L Decoder", ZImageI2LDecoder, load_i2l_decoder_weights)
    # Forward pass with concatenated embeddings
    dummy_emb = mx.concatenate([siglip2_out, dinov3_out], axis=-1)  # (1, 5632)
    print(f"\n  Concatenated embedding shape: {dummy_emb.shape}")
    t0 = time.time()
    lora = i2l(dummy_emb[0])  # Pass single embedding, not batched
    mx.eval(lora)
    t1 = time.time()
    print(f"  i2L decoder output: {len(lora)} LoRA weight pairs")
    print(f"  Time: {t1 - t0:.2f}s")

    # Print some LoRA weight names and shapes
    print("\n  Sample LoRA weights:")
    for i, (k, v) in enumerate(sorted(lora.items())):
        print(f"    {k}: {v.shape}")
        if i >= 5:
            print(f"    ... ({len(lora) - 6} more)")
            break

    print(f"\n{'=' * 60}")
    print("✅ All weight loading tests complete!")
