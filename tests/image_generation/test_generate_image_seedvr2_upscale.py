import pytest

from mflux.models.common.config.model_config import ModelConfig
from tests.image_generation.helpers.seedvr2_upscale_test_helper import SeedVR2UpscaleTestHelper


class TestSeedVR2Upscale:
    @pytest.mark.slow
    def test_seedvr2_upscale(self):
        SeedVR2UpscaleTestHelper.assert_matches_reference_image(
            reference_image_path="reference_seedvr2_upscaled.png",
            output_image_path="output_seedvr2_upscaled.png",
            input_image_path="low_res.jpg",
            seed=42,
            resolution=1080,
            quantize=8,
        )

    @pytest.mark.slow
    def test_seedvr2_7b_upscale(self):
        SeedVR2UpscaleTestHelper.assert_matches_reference_image(
            reference_image_path="reference_seedvr2_7b_upscaled.png",
            output_image_path="output_seedvr2_7b_upscaled.png",
            input_image_path="reference_z_image_turbo_lora.png",
            model_config=ModelConfig.seedvr2_7b(),
            seed=42,
            resolution=1080,
            quantize=8,
        )
