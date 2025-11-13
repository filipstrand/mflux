from datetime import datetime

from mflux.config.config import Config
from mflux.models.fibo.variants.txt2img.fibo import FIBO
from mflux_debugger._scripts.debug_txt2img_config import TXT2IMG_DEBUG_CONFIG
from mflux_debugger.image_archive import archive_images
from mflux_debugger.image_tensor_paths import get_images_latest_framework_dir


def main():
    config = TXT2IMG_DEBUG_CONFIG

    # Initialize FIBO model (minimal - VAE decoder only for now)
    model = FIBO(
        quantize=None,  # Turn off quantization for debugging
        local_path=None,
    )

    # Generate image (will load latents from PyTorch and decode with VAE)
    image = model.generate_image(
        seed=config.seed,
        prompt=config.prompt,
        negative_prompt=config.negative_prompt,
        config=Config(
            num_inference_steps=config.num_inference_steps,
            height=config.height,
            width=config.width,
            guidance=config.guidance,
            image_path=None,
            image_strength=None,
            scheduler=None,  # Not used yet (no denoising loop)
        ),
    )

    # Archive old images before saving new one (keep only latest)
    archive_images("mlx")

    # Save to images/latest/mlx/ directory
    images_dir = get_images_latest_framework_dir("mlx")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = images_dir / f"debug_mflux_txt2img_{timestamp}.png"
    image.save(path=str(output_path), export_json_metadata=False)
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
