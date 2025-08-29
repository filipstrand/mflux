#!/usr/bin/env python3
"""
Regression test for QwenImage Transformer functionality.
This test ensures that refactoring doesn't break the transformer forward pass.
"""

import mlx.core as mx
import numpy as np
from mflux.config.model_config import ModelConfig
from mflux.config.config import Config
from mflux.config.runtime_config import RuntimeConfig
from mflux.models.qwen.variants.txt2img.qwen_image import QwenImage


def test_qwen_transformer_regression():
    """
    Regression test for QwenImage Transformer forward pass.
    
    This test uses fixed seed and small inputs to quickly verify that 
    the transformer functionality produces consistent results. If any of the
    assertions fail, it indicates that the transformer behavior has changed.
    
    Test parameters:
    - Input latents: (1, 1024, 64) with seed=42
    - Text embeddings: (1, 77, 3584) with seed=43
    - Text mask: simple attention mask
    - Timestep: t=10 
    - Expected output: (1, 1024, 64) transformed latents
    """
    # Create QwenImage instance with quantization for faster loading
    qwen = QwenImage(
        model_config=ModelConfig.from_name("qwen-image"),
        quantize=6,
        local_path=None,
        lora_paths=None,
        lora_scales=None,
    )
    
    # Create test configuration for transformer
    config = Config(
        height=512,  # 32*16 = 512 (upscale factor 16)
        width=512,
        num_inference_steps=20,
        guidance=7.0,
    )
    test_config = RuntimeConfig(config, qwen.model_config)
    
    # Create reproducible test inputs (correct format for transformer)
    latent_height = 512 // 16  # 32
    latent_width = 512 // 16   # 32  
    test_latents = mx.random.normal(
        shape=(1, latent_height * latent_width, 64),  # (1, 1024, 64)
        key=mx.random.key(42)  # Fixed seed for reproducible results
    )
    
    # Create test text embeddings and mask
    test_encoder_hidden_states = mx.random.normal(
        shape=(1, 77, 3584),
        key=mx.random.key(43)  # Different seed for text
    )
    
    # Simple attention mask (all ones = all tokens are real)
    test_encoder_hidden_states_mask = mx.ones((1, 77), dtype=mx.float32)
    
    # Test timestep
    test_timestep = 10
    
    # Run transformer
    result = qwen.transformer(
        t=test_timestep,
        config=test_config,
        hidden_states=test_latents,
        encoder_hidden_states=test_encoder_hidden_states,
        encoder_hidden_states_mask=test_encoder_hidden_states_mask,
    )
    
    # Regression test assertions - based on baseline run
    EXPECTED_OUTPUT_SHAPE = (1, 1024, 64)  # (1, 32*32, 64) 
    EXPECTED_OUTPUT_MIN = -6.34730339050293
    EXPECTED_OUTPUT_MAX = 6.162752628326416
    EXPECTED_OUTPUT_MEAN = 0.05917155742645264
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
    result2 = qwen.transformer(
        t=test_timestep,
        config=test_config,
        hidden_states=test_latents,
        encoder_hidden_states=test_encoder_hidden_states,
        encoder_hidden_states_mask=test_encoder_hidden_states_mask,
    )
    assert mx.allclose(result, result2), "Transformer should be deterministic with same input"


if __name__ == "__main__":
    """Run the test directly for debugging/development."""
    print("🚀 Running QwenImage Transformer regression test...")
    try:
        test_qwen_transformer_regression()
        print("✅ Regression test passed!")
    except Exception as e:
        print("❌ Regression test failed!")
        print(f"Error: {e}")
        raise
