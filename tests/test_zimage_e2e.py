"""End-to-end tests that load actual model weights.

These require ~16GB+ memory and network access.
They require a Mac with 48GB RAM for unqantized models.
Skip by default: pytest -m "not high_memory_requirement"
"""

import pytest
from PIL import Image

pytestmark = pytest.mark.high_memory_requirement


class TestZImageEndToEnd:
    """End-to-end tests that load actual model weights.

    These tests require ~16GB+ memory and network access.
    Skip by default: pytest -m "not high_memory_requirement"
    """

    @pytest.mark.skipif(
        True,  # Skip until model is available and verified
        reason="Requires model download and high memory",
    )
    def test_e2e_generation_8bit(self):
        """Test actual generation with 8-bit quantization."""
        from mflux.config.config import Config
        from mflux.config.model_config import ModelConfig
        from mflux.zimage import ZImage

        zimage = ZImage(
            model_config=ModelConfig.zimage_turbo(),
            quantize=8,
        )

        config = Config(
            num_inference_steps=9,
            height=512,
            width=512,
            guidance=0.0,
        )

        image = zimage.generate_image(
            seed=42,
            prompt="A red apple on a white table",
            config=config,
        )

        assert hasattr(image, "image")
        assert isinstance(image.image, Image.Image)
        assert image.image.size == (512, 512)
        assert image.image.mode == "RGB"

    @pytest.mark.skipif(
        True,
        reason="Requires model download and high memory",
    )
    def test_e2e_all_quantization_levels(self):
        """Verify all quantization levels work."""
        from mflux.config.config import Config
        from mflux.config.model_config import ModelConfig
        from mflux.zimage import ZImage

        config = Config(
            num_inference_steps=2,  # Minimal steps
            height=256,
            width=256,
            guidance=0.0,
        )

        for bits in [8, 6, 4]:  # Skip 3-bit for speed
            zimage = ZImage(
                model_config=ModelConfig.zimage_turbo(),
                quantize=bits,
            )

            image = zimage.generate_image(
                seed=42,
                prompt="test",
                config=config,
            )

            assert isinstance(image.image, Image.Image)
