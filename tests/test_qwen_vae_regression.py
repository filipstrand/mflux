#!/usr/bin/env python3
"""
Regression test for QwenImage VAE decode functionality.
This test ensures that refactoring doesn't break the decode method.
"""

import mlx.core as mx
from mflux.config.model_config import ModelConfig
from mflux.models.qwen.variants.txt2img.qwen_image import QwenImage


def test_qwen_vae_decode_regression():
    """
    Regression test for QwenImage VAE decode method.
    
    This test uses a fixed seed and small input to quickly verify that 
    the decode functionality produces consistent results. If any of the
    assertions fail, it indicates that the decode behavior has changed.
    
    Test parameters:
    - Input: (1, 16, 32, 32) latents with seed=42
    - Expected output: (1, 3, 256, 256) image 
    - Scaling factor: 8x (32 -> 256)
    """
    # Create QwenImage instance with quantization for faster loading
    qwen = QwenImage(
        model_config=ModelConfig.from_name("qwen-image"),
        quantize=6,
        local_path=None,
        lora_paths=None,
        lora_scales=None,
    )
    
    # Create reproducible test input
    test_latents = mx.random.normal(
        shape=(1, 16, 32, 32), 
        key=mx.random.key(42)  # Fixed seed for reproducible results
    )
    
    # Run decode
    result = qwen.vae.decode(test_latents)
    
    # Regression test assertions - based on baseline run (with proper clamping)
    EXPECTED_OUTPUT_SHAPE = (1, 3, 256, 256)
    EXPECTED_OUTPUT_MIN = -1.0
    EXPECTED_OUTPUT_MAX = 1.0
    EXPECTED_OUTPUT_MEAN = -0.31647956371307373
    TOLERANCE = 1e-4
    
    # Basic checks
    assert result.shape == EXPECTED_OUTPUT_SHAPE, f"Expected shape {EXPECTED_OUTPUT_SHAPE}, got {result.shape}"
    assert result.dtype == mx.float32, f"Expected float32, got {result.dtype}"
    
    # Value checks for regression detection
    actual_min = float(mx.min(result).item())
    actual_max = float(mx.max(result).item()) 
    actual_mean = float(mx.mean(result).item())
    
    assert abs(actual_min - EXPECTED_OUTPUT_MIN) < TOLERANCE, f"Min changed: expected {EXPECTED_OUTPUT_MIN}, got {actual_min}"
    assert abs(actual_max - EXPECTED_OUTPUT_MAX) < TOLERANCE, f"Max changed: expected {EXPECTED_OUTPUT_MAX}, got {actual_max}"
    assert abs(actual_mean - EXPECTED_OUTPUT_MEAN) < TOLERANCE, f"Mean changed: expected {EXPECTED_OUTPUT_MEAN}, got {actual_mean}"
    
    # Determinism check
    result2 = qwen.vae.decode(test_latents)
    assert mx.allclose(result, result2), "Decode should be deterministic with same input"


def test_qwen_vae_encode_regression():
    """
    Regression test for QwenImage VAE encode method.
    
    This test uses a fixed seed and small input to quickly verify that 
    the encode functionality produces consistent results. If any of the
    assertions fail, it indicates that the encode behavior has changed.
    
    Test parameters:
    - Input: (1, 3, 256, 256) image with seed=123
    - Expected output: (1, 16, 32, 32) latents
    - Scaling factor: 1/8x (256 -> 32)
    """
    # Create QwenImage instance with quantization for faster loading
    qwen = QwenImage(
        model_config=ModelConfig.from_name("qwen-image"),
        quantize=6,
        local_path=None,
        lora_paths=None,
        lora_scales=None,
    )
    
    # Create reproducible test input - image-like tensor
    test_image = mx.random.normal(
        shape=(1, 3, 256, 256), 
        key=mx.random.key(123)  # Different seed from decode test
    )
    
    # Run encode
    result = qwen.vae.encode(test_image)
    
    # Regression test assertions - based on baseline run
    EXPECTED_OUTPUT_SHAPE = (1, 16, 32, 32)
    EXPECTED_OUTPUT_MIN = -3.2379164695739746
    EXPECTED_OUTPUT_MAX = 3.6191279888153076
    EXPECTED_OUTPUT_MEAN = 0.3937392830848694
    TOLERANCE = 1e-4
    
    # Basic checks
    assert result.shape == EXPECTED_OUTPUT_SHAPE, f"Expected shape {EXPECTED_OUTPUT_SHAPE}, got {result.shape}"
    assert result.dtype == mx.float32, f"Expected float32, got {result.dtype}"
    
    # Value checks for regression detection
    actual_min = float(mx.min(result).item())
    actual_max = float(mx.max(result).item()) 
    actual_mean = float(mx.mean(result).item())
    
    assert abs(actual_min - EXPECTED_OUTPUT_MIN) < TOLERANCE, f"Min changed: expected {EXPECTED_OUTPUT_MIN}, got {actual_min}"
    assert abs(actual_max - EXPECTED_OUTPUT_MAX) < TOLERANCE, f"Max changed: expected {EXPECTED_OUTPUT_MAX}, got {actual_max}"
    assert abs(actual_mean - EXPECTED_OUTPUT_MEAN) < TOLERANCE, f"Mean changed: expected {EXPECTED_OUTPUT_MEAN}, got {actual_mean}"
    
    # Determinism check
    result2 = qwen.vae.encode(test_image)
    assert mx.allclose(result, result2), "Encode should be deterministic with same input"


if __name__ == "__main__":
    """Run the tests directly for debugging/development."""
    print("🚀 Running QwenImage VAE regression tests...")
    
    try:
        print("  Testing decode...")
        test_qwen_vae_decode_regression()
        print("  ✅ Decode regression test passed!")
        
        print("  Testing encode...")
        test_qwen_vae_encode_regression()
        print("  ✅ Encode regression test passed!")
        
        print("🎉 All VAE regression tests passed!")
    except Exception as e:
        print("❌ Regression test failed!")
        print(f"Error: {e}")
        raise
