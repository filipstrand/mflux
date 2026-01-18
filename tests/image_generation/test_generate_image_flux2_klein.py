import pytest

from mflux.models.common.config import ModelConfig
from mflux.models.flux2.variants import Flux2Klein
from tests.image_generation.helpers.image_generation_test_helper import ImageGeneratorTestHelper


class TestImageGeneratorFlux2Klein:
    @pytest.mark.slow
    def test_flux2_klein_text_to_image(self):
        ImageGeneratorTestHelper.assert_matches_reference_image(
            reference_image_path="reference_flux2_klein_tiger_swim.png",
            output_image_path="output_flux2_klein_tiger_swim.png",
            model_class=Flux2Klein,
            model_config=ModelConfig.flux2_klein_4b(),
            quantize=None,
            steps=4,
            seed=5,
            height=1024,
            width=1024,
            guidance=1.0,
            prompt=(
                "Photorealistic tiger swimming in a jungle river, water droplets, golden hour, 85mm lens, natural light"
            ),
            mismatch_threshold=0.15,
        )

    @pytest.mark.slow
    def test_flux2_klein_9b_text_to_image(self):
        ImageGeneratorTestHelper.assert_matches_reference_image(
            reference_image_path="reference_flux2_klein_9b_tiger_swim.png",
            output_image_path="output_flux2_klein_9b_tiger_swim.png",
            model_class=Flux2Klein,
            model_config=ModelConfig.flux2_klein_9b(),
            quantize=None,
            steps=4,
            seed=5,
            height=1024,
            width=1024,
            guidance=1.0,
            prompt=(
                "Photorealistic tiger swimming in a jungle river, water droplets, golden hour, 85mm lens, natural light"
            ),
            mismatch_threshold=0.15,
        )
