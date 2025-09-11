#!/usr/bin/env python3
"""
Debug script for MFLUX Qwen Image Edit (hardcoded inputs, low-res)
"""

from mflux.config.config import Config
from mflux.models.qwen.variants.edit.qwen_image_edit import QwenImageEdit


def main():
    print("\n🔧 Debug: MFLUX Qwen Image Edit (MLX)\n" + "=" * 60)

    prompt = "make the person wear sunglasses, realistic, high quality"
    negative_prompt = ""
    image_path = "/Users/filipstrand/Desktop/mflux/person_no_glasses.png"

    width = 128
    height = 128
    steps = 4
    seed = 2
    guidance = 4.0

    print(f"Prompt: {prompt}")
    print(f"Negative: {negative_prompt!r}")
    print(f"Image: {image_path}")
    print(f"Resolution: {width}x{height}, steps={steps}, seed={seed}, guidance={guidance}")

    try:
        print("🚀 Loading QwenImageEdit (MLX)...")
        model = QwenImageEdit(
            quantize=None,
            local_path=None,
            lora_paths=None,
            lora_scales=None,
        )
        print("✅ Model loaded")

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
                image_path=image_path,
            ),
        )
        out = f"debug_mlx_qwen_edit_{width}x{height}_{steps}steps_seed{seed}.png"
        image.save(path=out)
        print(f"💾 Saved: {out}")
    except Exception as e:  # noqa: BLE001
        print(f"❌ Error: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
