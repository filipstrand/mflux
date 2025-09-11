#!/usr/bin/env python3
"""
Debug script for Diffusers Qwen Image Edit (CPU, low-res)
"""

import math
import os
import sys

import torch
from PIL import Image

# Add local diffusers to path
ROOT = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, os.path.join(ROOT, "diffusers", "src"))

from diffusers import DiffusionPipeline, FlowMatchEulerDiscreteScheduler  # noqa: E402


def main():
    print("\n🔧 Debug: Diffusers Qwen Image Edit (MPS)\n" + "=" * 60)

    prompt = "Remove the glasses from the person's face"
    negative_prompt = "ugly"
    image_path = "/Users/filipstrand/Desktop/person.png"

    width = 512
    height = 512
    steps = 3
    seed = 42
    guidance = 3.5
    device = "mps"  # Use MPS for faster execution on macOS
    torch_dtype = torch.float16  # Use float16 for MPS compatibility

    print(f"Prompt: {prompt}")
    print(f"Negative: {negative_prompt!r}")
    print(f"Image: {image_path}")
    print(f"Resolution: {width}x{height}, steps={steps}, seed={seed}, guidance={guidance}")

    try:
        print("⚙️ Setting up scheduler...")
        scheduler_config = {
            "base_image_seq_len": 256,
            "base_shift": math.log(3),
            "invert_sigmas": False,
            "max_image_seq_len": 8192,
            "max_shift": math.log(3),
            "num_train_timesteps": 1000,
            "shift": 1.0,
            "shift_terminal": None,
            "stochastic_sampling": False,
            "time_shift_type": "exponential",
            "use_beta_sigmas": False,
            "use_dynamic_shifting": True,
            "use_exponential_sigmas": False,
            "use_karras_sigmas": False,
        }
        scheduler = FlowMatchEulerDiscreteScheduler.from_config(scheduler_config)
        print("✅ Scheduler configured")

        print("🚀 Loading pipeline Qwen/Qwen-Image-Edit (SKIPPING transformer)...")
        # Skip transformer loading by passing transformer=None
        pipe = DiffusionPipeline.from_pretrained(
            "Qwen/Qwen-Image-Edit", 
            scheduler=scheduler, 
            torch_dtype=torch_dtype, 
            low_cpu_mem_usage=True,
            transformer=None  # 🔧 SKIP transformer to save memory!
        )
        pipe = pipe.to(device)
        print("✅ Pipeline loaded WITHOUT transformer - massive memory savings!")

        print("🎨 Starting generation to extract latents...")
        pil_image = Image.open(image_path).convert("RGB")
        print(f"🔎 Debug: Original PIL image size={pil_image.size}")
        
        # Run the pipeline to trigger latent preparation and saving
        try:
            image = pipe(
                image=pil_image,
                prompt=prompt,
                negative_prompt=negative_prompt,
                width=width,
                height=height,
                num_inference_steps=steps,
                true_cfg_scale=guidance,
                generator=torch.Generator(device=device).manual_seed(seed),
            ).images[0]
            print("🔎 Debug: Pipeline completed successfully")
        except KeyboardInterrupt:
            print("🔎 Debug: Stopped by user")
            return
        except Exception as e:
            print(f"🔎 Debug: Pipeline error: {e}")
            # Continue anyway, latents might have been saved
            print("🔎 Debug: Continuing to check for saved artifacts...")
            image = None
        
        if image is not None:
            out = f"debug_diffusers_qwen_edit_{width}x{height}_{steps}steps_seed{seed}.png"
            image.save(out)
            print(f"💾 Saved: {out}")
        else:
            print("🔎 Debug: No final image to save, but early artifacts should be saved")
    except Exception as e:  # noqa: BLE001
        print(f"❌ Error: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
