import json
import os
import tempfile
from datetime import datetime
from pathlib import Path

import pytest

from mflux.models.common.config import ModelConfig
from mflux.models.flux.variants.txt2img.flux import Flux1
from mflux.utils.metadata_reader import MetadataReader


class TestMetadata:
    @pytest.mark.slow
    def test_metadata_complete(self):
        # Use a temporary file - don't keep it open to avoid conflicts
        fd, temp_path = tempfile.mkstemp(suffix=".png")
        os.close(fd)  # Close the file descriptor immediately
        output_path = Path(temp_path)

        try:
            # Use img2img to test image_path and image_strength metadata fields
            reference_image = Path(__file__).parent.parent / "resources" / "reference_schnell.png"

            # Generate a small, fast image with schnell img2img (256x256, 2 steps, quantized)
            flux = Flux1(
                model_config=ModelConfig.schnell(),
                quantize=8,
            )

            image = flux.generate_image(
                seed=42,
                prompt="A simple test image",
                num_inference_steps=2,
                height=256,
                width=256,
                image_path=reference_image,
                image_strength=0.3,
            )

            # Save with metadata (overwrite=True since mkstemp creates an empty file)
            image.save(path=output_path, overwrite=True)

            # =================================================================
            # Test 1: Read metadata and verify structure
            # =================================================================
            metadata = MetadataReader.read_all_metadata(output_path)

            assert metadata is not None, "Metadata should not be None"
            assert "exif" in metadata, "EXIF metadata should be present"

            exif = metadata["exif"]

            # =================================================================
            # Test 2: Core generation parameters
            # =================================================================
            assert exif.get("seed") == 42, "Seed should match"
            assert exif.get("steps") == 2, "Steps should match"
            assert exif.get("prompt") == "A simple test image", "Prompt should match"
            assert exif.get("model") == "black-forest-labs/FLUX.1-schnell", "Model should match"

            # =================================================================
            # Test 3: Dimensions (NEW FEATURE)
            # =================================================================
            assert exif.get("width") == 256, "Width should be saved"
            assert exif.get("height") == 256, "Height should be saved"

            # =================================================================
            # Test 4: Technical parameters
            # =================================================================
            assert exif.get("quantize") == 8, "Quantization should match"
            assert exif.get("precision") is not None, "Precision should be set"

            # =================================================================
            # Test 5: MFLUX version (not hardcoded)
            # =================================================================
            assert exif.get("mflux_version") is not None, "MFLUX version should be present"
            assert exif.get("mflux_version") != "unknown", "MFLUX version should not be unknown"

            # =================================================================
            # Test 6: Generation time
            # =================================================================
            assert exif.get("generation_time_seconds") is not None, "Generation time should be present"
            assert isinstance(exif.get("generation_time_seconds"), (int, float)), "Generation time should be numeric"
            assert exif.get("generation_time_seconds") > 0, "Generation time should be positive"

            # =================================================================
            # Test 7: Creation timestamp (NEW FEATURE)
            # =================================================================
            assert exif.get("created_at") is not None, "Creation timestamp should be present"
            created_at = exif.get("created_at")

            # Verify it's valid ISO format
            dt = datetime.fromisoformat(created_at)
            now = datetime.now()
            time_diff = abs((now - dt).total_seconds())
            assert time_diff < 120, f"Timestamp should be recent (within 120s), but was {time_diff}s ago"

            # =================================================================
            # Test 8: Img2img-specific fields are populated
            # =================================================================
            assert exif.get("image_path") is not None, "Image path should be set for img2img"
            assert str(reference_image) in exif.get("image_path"), "Image path should contain reference image name"
            assert exif.get("image_strength") == 0.3, "Image strength should match config"

            # =================================================================
            # Test 9: Optional fields are None when not used
            # =================================================================
            assert exif.get("lora_paths") is None, "LoRA paths should be None when not used"
            assert exif.get("lora_scales") is None, "LoRA scales should be None when not used"
            assert exif.get("controlnet_image_path") is None, "ControlNet path should be None when not used"
            assert exif.get("negative_prompt") is None, "Negative prompt should be None when not used"
            assert exif.get("guidance") is None, "Guidance should be None for schnell model"

            # =================================================================
            # Test 10: EXIF JSON is valid and parseable
            # =================================================================
            import piexif
            from PIL import Image

            with Image.open(output_path) as img:
                exif_bytes = img.info.get("exif")
                assert exif_bytes is not None, "EXIF bytes should be present"

            exif_dict = piexif.load(exif_bytes)
            user_comment = exif_dict.get("Exif", {}).get(piexif.ExifIFD.UserComment)
            assert user_comment is not None, "UserComment should be present"

            # Decode JSON
            if user_comment.startswith(b"ASCII\x00\x00\x00"):
                json_str = user_comment[8:].decode("utf-8")
            else:
                json_str = user_comment.decode("utf-8")

            metadata_parsed = json.loads(json_str)
            # Just verify it's valid JSON
            assert metadata_parsed.get("prompt") == "A simple test image", "EXIF JSON should contain correct prompt"

            # =================================================================
            # Test 11: mflux-info command output
            # =================================================================
            from mflux.utils.info_util import InfoUtil

            output = InfoUtil.format_metadata(metadata)
            assert "A simple test image" in output, "Prompt should be in output"
            assert "MFLUX" in output, "MFLUX should be mentioned"
            assert "42" in output, "Seed should be in output"
            assert "256" in output, "Dimensions should be in output"
            assert "Generation Time:" in output, "Generation time should be shown"
            assert "Created:" in output, "Creation timestamp should be shown"
            assert "Source Image:" in output, "Source image should be shown for img2img"
            assert "Image Strength:" in output, "Image strength should be shown for img2img"

            # =================================================================
            # Test 12: Metadata reader handles nonexistent files
            # =================================================================
            nonexistent_metadata = MetadataReader.read_all_metadata(Path("/nonexistent/file.png"))
            assert nonexistent_metadata.get("exif") is None, "Nonexistent file should return None for EXIF"
            assert nonexistent_metadata.get("xmp") is None, "Nonexistent file should return None for XMP"

        finally:
            # Cleanup: Always remove the temporary file
            if output_path.exists():
                output_path.unlink()
