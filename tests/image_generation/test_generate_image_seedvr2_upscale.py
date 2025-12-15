import pytest

from mflux.models.common.vae.tiling_config import TilingConfig
from tests.image_generation.helpers.seedvr2_upscale_test_helper import SeedVR2UpscaleTestHelper


class TestSeedVR2Upscale:
    @pytest.mark.slow
    def test_seedvr2_upscale(self):
        SeedVR2UpscaleTestHelper.assert_matches_reference_image(
            reference_image_path="reference_seedvr2_upscaled.png",
            output_image_path="output_seedvr2_upscaled.png",
            input_image_path="low_res.jpg",
            seed=42,
            resolution=384,
            quantize=None,
        )

    @pytest.mark.slow
    def test_seedvr2_upscale_tiled(self):
        SeedVR2UpscaleTestHelper.assert_matches_reference_image(
            reference_image_path="reference_seedvr2_upscaled_tiled.png",
            output_image_path="output_seedvr2_upscaled_tiled.png",
            input_image_path="low_res.jpg",
            seed=42,
            resolution=1080,
            quantize=4,
            tiling_config=TilingConfig(),
        )
