import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

import torch
from diffusers.pipelines.qwenimage import QwenImagePipeline

from mflux_debugger._scripts.debug_txt2img_config import TXT2IMG_DEBUG_CONFIG
from mflux_debugger.image_archive import archive_images
from mflux_debugger.image_tensor_paths import get_images_latest_framework_dir


def main():
    model_name = "Qwen/Qwen-Image"
    config = TXT2IMG_DEBUG_CONFIG

    pipe = QwenImagePipeline.from_pretrained(model_name, torch_dtype=torch.bfloat16)

    try:
        pipe = pipe.to("mps")
        device = "mps"
    except RuntimeError:
        pipe = pipe.to("cpu")
        device = "cpu"

    # Generate initial latents with seed
    generator = torch.Generator(device=device).manual_seed(config.seed)

    # Create latents shape for txt2img: [batch, seq_len, channels]
    # For 512x512 image: 512/16 = 32, 32*32 = 1024 tokens
    latent_height = config.height // 16
    latent_width = config.width // 16
    seq_len = latent_height * latent_width
    latents = torch.randn(
        (1, seq_len, 64),  # [batch, seq_len, channels]
        generator=generator,
        device=device,
        dtype=torch.bfloat16,
    )

    image = pipe(
        prompt=config.prompt,
        negative_prompt=config.negative_prompt,
        height=config.height,
        width=config.width,
        num_inference_steps=config.num_inference_steps,
        true_cfg_scale=config.guidance,
        generator=generator,
        latents=latents,  # Pass pre-computed latents
    ).images[0]

    # Archive old images before saving new one (keep only latest)
    archive_images("pytorch")

    # Save to images/latest/pytorch/ directory
    images_dir = get_images_latest_framework_dir("pytorch")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = images_dir / f"debug_diffusers_txt2img_{timestamp}.png"
    image.save(str(output_path))
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
