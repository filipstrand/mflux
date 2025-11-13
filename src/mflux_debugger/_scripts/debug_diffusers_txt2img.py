import json
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

import torch
from diffusers import BriaFiboPipeline
from diffusers.modular_pipelines import ModularPipeline

from mflux_debugger._scripts.debug_txt2img_config import TXT2IMG_DEBUG_CONFIG
from mflux_debugger.image_archive import archive_images
from mflux_debugger.image_tensor_paths import get_images_latest_framework_dir


def get_default_negative_prompt(existing_json: dict) -> str:
    """Generate default negative prompt based on JSON style."""
    negative_prompt = ""
    style_medium = existing_json.get("style_medium", "").lower()
    if style_medium in ["photograph", "photography", "photo"]:
        negative_prompt = """{'style_medium':'digital illustration','artistic_style':'non-realistic'}"""
    return negative_prompt


def main():
    config = TXT2IMG_DEBUG_CONFIG
    torch.set_grad_enabled(False)

    # -------------------------------
    # Load the VLM pipeline
    # -------------------------------
    # Using local VLM
    vlm_pipe = ModularPipeline.from_pretrained("briaai/FIBO-VLM-prompt-to-JSON", trust_remote_code=True)
    # Using Gemini API, requires GOOGLE_API_KEY environment variable
    # assert os.getenv("GOOGLE_API_KEY") is not None, "GOOGLE_API_KEY environment variable is not set"
    # vlm_pipe = ModularPipeline.from_pretrained("briaai/FIBO-gemini-prompt-to-JSON", trust_remote_code=True)

    # -------------------------------
    # Load the FIBO pipeline
    # -------------------------------
    pipe = BriaFiboPipeline.from_pretrained("briaai/FIBO", torch_dtype=torch.bfloat16)

    # Try to use MPS (Mac), fallback to CPU
    try:
        pipe = pipe.to("mps")
        device = "mps"
    except RuntimeError:
        pipe = pipe.to("cpu")
        device = "cpu"
    # Uncomment if you're getting CUDA OOM errors
    # pipe.enable_model_cpu_offload()

    # -------------------------------
    # Run Prompt to JSON
    # -------------------------------
    # Create a prompt to generate an initial image
    output = vlm_pipe(prompt=config.prompt)
    json_prompt_generate = output.values["json_prompt"]

    # Get negative prompt based on JSON style, fallback to config if empty
    json_negative_prompt = get_default_negative_prompt(json.loads(json_prompt_generate))
    # Use config negative_prompt if JSON-based one is empty, otherwise use JSON-based one
    negative_prompt = json_negative_prompt if json_negative_prompt else config.negative_prompt

    # -------------------------------
    # Run Image Generation
    # -------------------------------
    # Generate the image from the structured json prompt
    generator = torch.Generator(device=device).manual_seed(config.seed)
    results_generate = pipe(
        prompt=json_prompt_generate,
        height=config.height,
        width=config.width,
        num_inference_steps=config.num_inference_steps,
        guidance_scale=config.guidance,
        negative_prompt=negative_prompt,
        generator=generator,
    )

    image = results_generate.images[0]

    # Archive old images before saving new one (keep only latest)
    archive_images("pytorch")

    # Save to images/latest/pytorch/ directory
    images_dir = get_images_latest_framework_dir("pytorch")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = images_dir / f"debug_diffusers_fibo_{timestamp}.png"
    image.save(str(output_path))
    print(f"Saved: {output_path}")

    # Save JSON prompt
    json_path = images_dir / f"debug_diffusers_fibo_{timestamp}_json_prompt.json"
    with open(json_path, "w") as f:
        f.write(json_prompt_generate)
    print(f"Saved JSON prompt: {json_path}")


if __name__ == "__main__":
    main()
