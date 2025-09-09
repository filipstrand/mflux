#!/usr/bin/env python3
"""
Debug script for MFLUX QwenImage with LoRA
Hardcoded parameters for development testing
"""

from mflux.config.config import Config
from mflux.config.model_config import ModelConfig
from mflux.models.qwen.variants.txt2img.qwen_image import QwenImage


def main():
    print("🔧 Debug: MFLUX QwenImage with LoRA")
    print("=" * 50)

    # Hardcoded parameters (matching your typical usage)
    prompt = "a majestic owl, ultra sharp, 4k, HDR"
    negative_prompt = "ugly, low-res, blurry"
    lora_path = "/Users/filipstrand/Library/Caches/mflux/loras/Qwen-Image-Lightning-4steps-V1.0.safetensors"

    # Low resolution for fast testing
    width = 128
    height = 128
    steps = 4
    seed = 2
    guidance = 4.0

    print(f"Prompt: {prompt}")
    print(f"Negative Prompt: {negative_prompt}")
    print(f"LoRA Path: {lora_path}")
    print(f"Resolution: {width}x{height}")
    print(f"Steps: {steps}")
    print(f"Seed: {seed}")
    print(f"Guidance: {guidance}")
    print()

    try:
        # 1. Initialize the QwenImage model with LoRA
        print("🚀 Loading QwenImage model with LoRA...")
        model = QwenImage(
            model_config=ModelConfig.from_name(model_name="qwen"),
            quantize=None,  # Skip quantization for now as requested
            local_path=None,
            lora_paths=[lora_path],
            lora_scales=[1.0],  # Default LoRA scale
        )
        print("✅ Model loaded successfully")

        # 2. Generate image
        print("🎨 Generating image...")
        image = model.generate_image(
            seed=seed,
            prompt=prompt,
            negative_prompt=negative_prompt,
            config=Config(
                num_inference_steps=steps,
                height=height,
                width=width,
                guidance=guidance,
            ),
        )
        print("✅ Image generated successfully")

        # 3. Save the image
        output_path = f"debug_mflux_qwen_lora_{width}x{height}_{steps}steps_seed{seed}.png"
        image.save(path=output_path)
        print(f"💾 Image saved: {output_path}")

    except Exception as e:  # noqa: BLE001
        print(f"❌ Error: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
