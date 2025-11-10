import os
from datetime import datetime
from pathlib import Path

os.environ["HF_HOME"] = os.path.expanduser("~/.cache/huggingface")

from mflux.config.config import Config
from mflux.models.qwen.variants.edit.qwen_image_edit_plus import QwenImageEditPlus
from mflux_debugger.image_archive import archive_images
from mflux_debugger.image_tensor_paths import get_images_latest_framework_dir


def main():
    # Apply Slytherin wallpaper to coffee cup
    project_root = Path(__file__).parent.parent.parent.parent
    coffee_cup_path = project_root / "coffee_cup_clean.png"
    wallpaper_path = project_root / "slytherin_wallpaper_output.png"

    # Increase resolution by 50%: 1024x688 -> 1536x1040
    target_width = 1536
    target_height = 1040

    prompt = (
        "Apply the Slytherin wallpaper pattern and colors from the second image onto the coffee cup "
        "in the first image, replacing the brown cardboard areas with the wallpaper design. "
        "CRITICAL: Preserve ONLY the physical texture and structure of the paper/cardboard cup - "
        "the tactile grip ring (raised horizontal stripe texture bands), paper surface texture, "
        "and all surface details. However, replace the brown cardboard color with the wallpaper's "
        "silver and emerald green color scheme. The cup should display the Slytherin logo and gold "
        "star trail in the wallpaper's colors (silver, emerald green, gold), not the original brown. "
        "The wallpaper pattern should wrap around the coffee cup following its cylindrical shape "
        "and curvature, creating a realistic 3D effect with the pattern naturally bending and "
        "conforming to the cup's form and grip ring texture. Preserve all lighting, shadows, highlights, "
        "and reflections that show the paper texture, but apply the wallpaper's silver and emerald green "
        "colors instead of brown. The grip ring stripes should remain visible and maintain their raised "
        "appearance, with the wallpaper pattern and colors flowing over them naturally. Maintain the exact "
        "shape, dimensions, and perspective of the coffee cup. The background, table surface, and all "
        "other elements should remain completely unchanged. The Slytherin logo and gold star trail should "
        "be clearly visible on the cup in their proper colors, properly distorted to match the cup's "
        "cylindrical perspective while respecting the existing paper texture and grip ring."
    )

    negative_prompt = (
        "ugly, blurry, low quality, distorted, deformed, pixelated, artifacts, noise, "
        "flat pattern, pattern not wrapping, incorrect perspective, wrong curvature, "
        "coffee cup shape changed, cup dimensions altered, cup structure redone, "
        "texture changed, grip ring removed, tactile texture lost, ceramic texture, "
        "glossy surface, shiny material, brown cardboard color, tan color, beige color, "
        "unrealistic lighting, missing shadows, pattern floating, wallpaper not conforming to cup shape, "
        "background changed, table changed, other objects modified, smooth surface, textureless"
    )

    model = QwenImageEditPlus(quantize=6)

    config = Config(
        height=target_height,
        width=target_width,
        num_inference_steps=30,
        guidance=4.0,
        image_path=str(coffee_cup_path),
        scheduler="flow_match_euler_discrete",
    )

    result = model.generate_image(
        prompt=prompt,
        seed=42,
        config=config,
        negative_prompt=negative_prompt,
        image_paths=[str(coffee_cup_path), str(wallpaper_path)],
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
