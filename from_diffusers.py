import os
import sys

# Add the local diffusers library to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "diffusers", "src"))

import torch

from diffusers import DiffusionPipeline

model_name = "Qwen/Qwen-Image"

# Load the pipeline (memory-friendly defaults)
torch_dtype = torch.bfloat16
device = "cpu"

pipe = DiffusionPipeline.from_pretrained(model_name, torch_dtype=torch_dtype, low_cpu_mem_usage=True)
pipe = pipe.to(device)
try:
    pipe.enable_attention_slicing()
except Exception:
    pass

# Generate image
prompt = '''A coffee shop entrance features a chalkboard sign reading "Qwen Coffee 😊 $2 per cup," with a neon light beside it displaying "通义千问". Next to it hangs a poster showing a beautiful Chinese woman, and beneath the poster is written "π≈3.1415926-53589793-23846264-33832795-02384197". Ultra HD, 4K, cinematic composition'''
negative_prompt = " "

width = 512
height = 512
steps = 4
seed = 44

image = pipe(
    prompt=prompt,
    negative_prompt=negative_prompt,
    width=width,
    height=height,
    num_inference_steps=steps,
    true_cfg_scale=4.0,
    generator=torch.Generator(device=device).manual_seed(seed),
).images[0]

out_path = f"example_{width}x{height}_{steps}steps_seed{seed}.png"
image.save(out_path)
print(f"Saved {out_path}")
