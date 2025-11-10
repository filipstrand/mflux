import os
from datetime import datetime

os.environ["HF_HOME"] = os.path.expanduser("~/.cache/huggingface")

from mflux.config.config import Config
from mflux.models.qwen.variants.edit.qwen_image_edit import QwenImageEdit
from mflux_debugger._scripts.debug_edit_config import EDIT_DEBUG_CONFIG
from mflux_debugger.image_archive import archive_images
from mflux_debugger.image_tensor_paths import get_images_latest_framework_dir


def main():
    config_values = EDIT_DEBUG_CONFIG

    model = QwenImageEdit()

    # Use the dog example from config - make it into a Dalmatian
    config = Config(
        height=config_values.height,
        width=config_values.width,
        num_inference_steps=config_values.num_inference_steps,
        guidance=config_values.guidance,
        image_path=config_values.image_path,
        scheduler="flow_match_euler_discrete",  # Use FlowMatchEulerDiscreteScheduler
    )

    result = model.generate_image(
        prompt=config_values.prompt,
        seed=config_values.seed,
        config=config,
        negative_prompt=config_values.negative_prompt,
    )

    # Archive old images before saving new one (keep only latest)
    archive_images("mlx")

    # Save to images/latest/mlx/ directory
    images_dir = get_images_latest_framework_dir("mlx")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = images_dir / f"debug_mflux_edit_output_{timestamp}.png"
    result.save(str(output_path))
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
