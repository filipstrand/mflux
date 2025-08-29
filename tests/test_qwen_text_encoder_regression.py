#!/usr/bin/env python3
"""
Regression test for QwenImage Text Encoder functionality.
This test ensures that refactoring doesn't break the text encoder forward pass.
"""

import mlx.core as mx
from mflux.config.model_config import ModelConfig
from mflux.models.qwen.variants.txt2img.qwen_image import QwenImage


def test_qwen_text_encoder_regression():
    """
    Regression test for QwenImage Text Encoder forward pass.
    
    This test uses fixed input to quickly verify that 
    the text encoder functionality produces consistent results. If any of the
    assertions fail, it indicates that the text encoder behavior has changed.
    
    Test parameters:
    - Input prompt: "a beautiful sunset over mountains"
    - Expected output: text embeddings and attention mask
    """
    # Create QwenImage instance with quantization for faster loading
    qwen = QwenImage(
        model_config=ModelConfig.from_name("qwen-image"),
        quantize=6,
        local_path=None,
        lora_paths=None,
        lora_scales=None,
    )
    
    # Test prompt
    test_prompt = "a beautiful sunset over mountains"
    
    # Encode the prompt (this exercises the text encoder)
    from mflux.models.qwen.model.qwen_text_encoder.qwen_prompt_encoder import QwenPromptEncoder
    
    prompt_embeds, prompt_mask, negative_prompt_embeds, negative_prompt_mask = QwenPromptEncoder.encode_prompt(
        prompt=test_prompt,
        negative_prompt=None,
        prompt_cache=qwen.prompt_cache,
        qwen_tokenizer=qwen.qwen_tokenizer,
        qwen_text_encoder=qwen.text_encoder,
    )
    
    # Regression test assertions - based on baseline run
    # Updated baseline after aligning implementation to reference behavior
    EXPECTED_PROMPT_EMBEDS_SHAPE = (1, 10, 3584)
    EXPECTED_PROMPT_MASK_SHAPE = (1, 10)
    EXPECTED_PROMPT_EMBEDS_MIN = -126.0
    EXPECTED_PROMPT_EMBEDS_MAX = 104.5
    EXPECTED_PROMPT_EMBEDS_MEAN = -0.10693359375
    TOLERANCE = 1e-4
    
    # Basic checks
    assert prompt_embeds.shape == EXPECTED_PROMPT_EMBEDS_SHAPE, f"Expected prompt embeds shape {EXPECTED_PROMPT_EMBEDS_SHAPE}, got {prompt_embeds.shape}"
    assert prompt_mask.shape == EXPECTED_PROMPT_MASK_SHAPE, f"Expected prompt mask shape {EXPECTED_PROMPT_MASK_SHAPE}, got {prompt_mask.shape}"
    assert prompt_embeds.dtype == mx.bfloat16, f"Expected bfloat16, got {prompt_embeds.dtype}"
    
    # Print actual values to get baseline for regression test
    actual_min = float(mx.min(prompt_embeds).item())
    actual_max = float(mx.max(prompt_embeds).item()) 
    actual_mean = float(mx.mean(prompt_embeds).item())
    
    print(f"🔍 Text encoder output for baseline:")
    print(f"   Prompt embeds shape: {prompt_embeds.shape}")
    print(f"   Prompt mask shape: {prompt_mask.shape}")
    print(f"   Prompt embeds dtype: {prompt_embeds.dtype}")
    print(f"   Prompt embeds min: {actual_min}")
    print(f"   Prompt embeds max: {actual_max}")
    print(f"   Prompt embeds mean: {actual_mean}")
    
    # Baseline values are set above from first run
    
    # Value checks for regression detection
    assert abs(actual_min - EXPECTED_PROMPT_EMBEDS_MIN) < TOLERANCE, f"Min changed: expected {EXPECTED_PROMPT_EMBEDS_MIN}, got {actual_min}"
    assert abs(actual_max - EXPECTED_PROMPT_EMBEDS_MAX) < TOLERANCE, f"Max changed: expected {EXPECTED_PROMPT_EMBEDS_MAX}, got {actual_max}"
    assert abs(actual_mean - EXPECTED_PROMPT_EMBEDS_MEAN) < TOLERANCE, f"Mean changed: expected {EXPECTED_PROMPT_EMBEDS_MEAN}, got {actual_mean}"
    
    # Determinism check - same prompt should give same results
    prompt_embeds2, prompt_mask2, _, _ = QwenPromptEncoder.encode_prompt(
        prompt=test_prompt,
        negative_prompt=None,
        prompt_cache=qwen.prompt_cache,
        qwen_tokenizer=qwen.qwen_tokenizer,
        qwen_text_encoder=qwen.text_encoder,
    )
    assert mx.allclose(prompt_embeds, prompt_embeds2), "Text encoder should be deterministic with same input"
    assert mx.allclose(prompt_mask, prompt_mask2), "Text encoder mask should be deterministic with same input"


if __name__ == "__main__":
    """Run the test directly for debugging/development."""
    print("🚀 Running QwenImage Text Encoder regression test...")
    try:
        test_qwen_text_encoder_regression()
        print("✅ Regression test passed!")
    except Exception as e:
        print("❌ Regression test failed!")
        print(f"Error: {e}")
        raise
