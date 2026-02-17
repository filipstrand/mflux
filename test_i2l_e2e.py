"""End-to-end test: encode style images into LoRA and verify output.

Run: python test_i2l_e2e.py

Downloads sample style images from DiffSynth-Studio/Z-Image-i2L,
encodes them into LoRA weights, and saves the result.
"""

import sys

sys.path.insert(0, "src")

from pathlib import Path

from huggingface_hub import hf_hub_download
from PIL import Image

from mflux.models.z_image.model.z_image_i2l.i2l_pipeline import ZImageI2LPipeline


def download_sample_images(num_images: int = 4) -> list[Image.Image]:
    """Download sample style images from the i2L repo."""
    images = []
    for i in range(num_images):
        path = hf_hub_download(
            repo_id="DiffSynth-Studio/Z-Image-i2L",
            filename=f"assets/style/1/{i}.jpg",
        )
        img = Image.open(path).convert("RGB")
        print(f"  Sample image {i}: {img.size[0]}x{img.size[1]}")
        images.append(img)
    return images


if __name__ == "__main__":
    print("Z-Image i2L End-to-End Test")
    print("=" * 60)

    # Download sample images
    print("\nDownloading sample style images...")
    images = download_sample_images(4)

    # Create pipeline
    print("\nCreating i2L pipeline...")
    pipeline = ZImageI2LPipeline.from_pretrained()

    # Generate LoRA
    output_path = "test_style_lora.safetensors"
    print(f"\nGenerating LoRA from {len(images)} images...")
    pipeline.generate_lora(images=images, output_path=output_path)

    # Verify output
    if Path(output_path).exists():
        size_mb = Path(output_path).stat().st_size / (1024 * 1024)
        print(f"\n✅ LoRA file created: {output_path} ({size_mb:.1f} MB)")

        # Quick inspect
        from safetensors import safe_open

        f = safe_open(output_path, framework="pt")
        keys = sorted(f.keys())
        print(f"  Total weight tensors: {len(keys)}")
        print("  Sample keys:")
        for k in keys[:5]:
            print(f"    {k}: {list(f.get_tensor(k).shape)}")
    else:
        print("\n❌ LoRA file was not created!")
        sys.exit(1)

    print("\n✅ End-to-end test complete!")
