from datetime import datetime

from mflux.config.config import Config
from mflux.config.model_config import ModelConfig
from mflux.models.qwen.variants.txt2img.qwen_image import QwenImage
from mflux_debugger._scripts.debug_txt2img_config import TXT2IMG_DEBUG_CONFIG
from mflux_debugger.image_archive import archive_images
from mflux_debugger.image_tensor_paths import get_images_latest_framework_dir


def main():
    model_name = "qwen"
    config = TXT2IMG_DEBUG_CONFIG

    model = QwenImage(
        model_config=ModelConfig.from_name(model_name=model_name),
        quantize=None,  # Turn off quantization for debugging
        local_path=None,
        lora_paths=None,
        lora_scales=None,
    )

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
            scheduler="flow_match_euler_discrete",  # Use FlowMatchEulerDiscreteScheduler
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
