import json
import random
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

import numpy as np
import torch
from diffusers import BriaFiboPipeline
from diffusers.modular_pipelines import ModularPipeline

from mflux_debugger._scripts.debug_txt2img_config import TXT2IMG_DEBUG_CONFIG
from mflux_debugger.image_archive import archive_images
from mflux_debugger.image_tensor_paths import get_images_latest_framework_dir
from mflux_debugger.semantic_checkpoint import debug_checkpoint
from mflux_debugger.tensor_debug import debug_save


def get_default_negative_prompt(existing_json: dict) -> str:
    """Generate default negative prompt based on JSON style."""
    negative_prompt = ""
    style_medium = existing_json.get("style_medium", "").lower()
    if style_medium in ["photograph", "photography", "photo"]:
        negative_prompt = """{'style_medium':'digital illustration','artistic_style':'non-realistic'}"""
    return negative_prompt


def main():
    # Clean up all debug files from previous runs
    # DISABLED during debugging - we need to preserve tensors for comparison
    # debug_full_cleanup()

    config = TXT2IMG_DEBUG_CONFIG
    torch.set_grad_enabled(False)

    # Lock all randomness sources for reproducibility
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    random.seed(config.seed)
    # Set deterministic mode for CUDA operations (if available)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

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

    # Use CPU offload to manage MPS memory constraints (like the working example)
    pipe.enable_model_cpu_offload()
    device = "mps"  # CPU offload still uses MPS for computation, just manages memory better

    # -------------------------------
    # Run Prompt to JSON
    # -------------------------------
    debug_checkpoint("before_vlm_prompt_to_json", metadata={"prompt": config.prompt}, skip=True)

    # Create a prompt to generate an initial image
    output = vlm_pipe(prompt=config.prompt)
    json_prompt_generate = output.values["json_prompt"]

    debug_checkpoint(
        "after_vlm_prompt_to_json",
        metadata={"json_prompt": json_prompt_generate, "original_prompt": config.prompt},
        skip=True,
    )

    # Get negative prompt based on JSON style, fallback to config if empty
    json_negative_prompt = get_default_negative_prompt(json.loads(json_prompt_generate))
    # Use config negative_prompt if JSON-based one is empty, otherwise use JSON-based one
    negative_prompt = json_negative_prompt if json_negative_prompt else config.negative_prompt

    debug_checkpoint(
        "before_pipeline_call",
        metadata={
            "json_prompt": json_prompt_generate,
            "negative_prompt": negative_prompt,
            "height": config.height,
            "width": config.width,
            "num_inference_steps": config.num_inference_steps,
            "guidance_scale": config.guidance,
            "seed": config.seed,
        },
        skip=True,
    )

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

    # Save VAE input latents (final latents before VAE decoding)
    # Note: BriaFiboPipelineOutput should contain latents
    if hasattr(results_generate, "latents") and results_generate.latents is not None:
        vae_input_latents = results_generate.latents
        debug_save(vae_input_latents, "vae_input_latents")
        debug_checkpoint(
            "after_pipeline_before_vae",
            metadata={
                "vae_input_shape": list(vae_input_latents.shape),
                "vae_input_dtype": str(vae_input_latents.dtype),
            },
        )
    else:
        debug_checkpoint("after_pipeline_before_vae", metadata={"note": "latents not available in output"})

    image = results_generate.images[0]

    # Save final image as tensor for comparison
    # Convert PIL image to tensor
    image_tensor = torch.from_numpy(np.array(image)).permute(2, 0, 1).float() / 255.0  # HWC -> CHW, normalize
    debug_save(image_tensor, "final_image_tensor")

    debug_checkpoint(
        "after_vae_decoding",
        metadata={
            "image_shape": list(image_tensor.shape),
            "image_dtype": str(image_tensor.dtype),
        },
    )

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
