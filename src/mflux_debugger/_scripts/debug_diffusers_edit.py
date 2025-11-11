import os
from datetime import datetime

os.environ["HF_HOME"] = os.path.expanduser("~/.cache/huggingface")

import torch
from diffusers.pipelines.qwenimage import QwenImageEditPlusPipeline
from PIL import Image

from mflux_debugger._scripts.debug_edit_config import EDIT_DEBUG_CONFIG
from mflux_debugger.image_archive import archive_images
from mflux_debugger.image_tensor_paths import get_images_latest_framework_dir


def main():
    model_name = "Qwen/Qwen-Image-Edit-2509"
    config = EDIT_DEBUG_CONFIG

    # Load single image for processor comparison
    image = Image.open(config.image_path).convert("RGB")
    images = [image]

    # Use the new edit plus pipeline (Qwen-Image-Edit-2509 model)
    pipe = QwenImageEditPlusPipeline.from_pretrained(model_name, torch_dtype=torch.bfloat16)

    try:
        pipe = pipe.to("mps")
        device = "mps"
    except RuntimeError:
        pipe = pipe.to("cpu")
        device = "cpu"

    generator = torch.Generator(device=device).manual_seed(config.seed)
    result = pipe(
        prompt=config.prompt,
        image=images,  # Pass list of images
        num_inference_steps=config.num_inference_steps,
        true_cfg_scale=config.guidance,
        generator=generator,
        negative_prompt=config.negative_prompt,  # Add negative prompt to enable CFG
        height=config.height,  # Use non-square: 384x512
        width=config.width,
    ).images[0]

    # Archive old images before saving new one (keep only latest)
    archive_images("pytorch")

    # Save to images/latest/pytorch/ directory
    images_dir = get_images_latest_framework_dir("pytorch")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = images_dir / f"debug_diffusers_edit_output_{timestamp}.png"
    result.save(str(output_path))
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
