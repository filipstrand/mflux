#!/usr/bin/env python3
"""
Debug script for Diffusers QwenImage with LoRA
Hardcoded parameters for development testing
"""

import math
import os
import sys

import torch

# Add the local diffusers library to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "diffusers", "src"))

from diffusers import DiffusionPipeline, FlowMatchEulerDiscreteScheduler


def main():
    print("🔧 Debug: Diffusers QwenImage with LoRA")
    print("=" * 50)

    # Hardcoded parameters (matching your typical usage, but CPU-friendly)
    prompt = "a majestic owl, ultra sharp, 4k, HDR"
    negative_prompt = "ugly, low-res, blurry"
    lora_repo = "lightx2v/Qwen-Image-Lightning"
    lora_weight_name = "Qwen-Image-Lightning-4steps-V1.0.safetensors"

    # Low resolution and CPU-friendly settings
    width = 128
    height = 128
    steps = 4
    seed = 2
    guidance = 1.0  # Lower guidance for LoRA as per your diffusers example
    device = "cpu"  # CPU mode for development
    torch_dtype = torch.bfloat16

    print(f"Prompt: {prompt}")
    print(f"Negative Prompt: {negative_prompt}")
    print(f"LoRA Repo: {lora_repo}")
    print(f"LoRA Weight: {lora_weight_name}")
    print(f"Resolution: {width}x{height}")
    print(f"Steps: {steps}")
    print(f"Seed: {seed}")
    print(f"Guidance: {guidance}")
    print(f"Device: {device}")
    print(f"Dtype: {torch_dtype}")
    print()

    try:
        # 1. Setup scheduler (from your diffusers example)
        print("⚙️ Setting up scheduler...")
        scheduler_config = {
            "base_image_seq_len": 256,
            "base_shift": math.log(3),  # We use shift=3 in distillation
            "invert_sigmas": False,
            "max_image_seq_len": 8192,
            "max_shift": math.log(3),  # We use shift=3 in distillation
            "num_train_timesteps": 1000,
            "shift": 1.0,
            "shift_terminal": None,  # set shift_terminal to None
            "stochastic_sampling": False,
            "time_shift_type": "exponential",
            "use_beta_sigmas": False,
            "use_dynamic_shifting": True,
            "use_exponential_sigmas": False,
            "use_karras_sigmas": False,
        }
        scheduler = FlowMatchEulerDiscreteScheduler.from_config(scheduler_config)
        print("✅ Scheduler configured")

        # 2. Load the pipeline
        print("🚀 Loading Diffusers QwenImage pipeline...")
        pipe = DiffusionPipeline.from_pretrained(
            "Qwen/Qwen-Image", scheduler=scheduler, torch_dtype=torch_dtype, low_cpu_mem_usage=True
        )
        pipe = pipe.to(device)

        # Enable memory optimizations
        try:
            pipe.enable_attention_slicing()
        except Exception:  # noqa: S110
            pass

        print("✅ Pipeline loaded successfully")

        # 3. Load LoRA weights
        print(f"📦 Loading LoRA weights from {lora_repo}...")
        pipe.load_lora_weights(lora_repo, weight_name=lora_weight_name)
        print("✅ LoRA weights loaded successfully")

        # 4. Generate image
        print("🎨 Generating image...")
        image = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            width=width,
            height=height,
            num_inference_steps=steps,
            true_cfg_scale=guidance,
            generator=torch.Generator(device=device).manual_seed(seed),
        ).images[0]
        print("✅ Image generated successfully")

        # 5. Save the image
        output_path = f"debug_diffusers_qwen_lora_{width}x{height}_{steps}steps_seed{seed}.png"
        image.save(output_path)
        print(f"💾 Image saved: {output_path}")

    except Exception as e:  # noqa: BLE001
        print(f"❌ Error: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
