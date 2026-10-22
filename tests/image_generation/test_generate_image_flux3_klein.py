import pytest

from mflux.models.common.config import ModelConfig
from mflux.models.flux3.variants import Flux3Klein
from tests.image_generation.helpers.image_generation_test_helper import ImageGeneratorTestHelper

FOX_PROMPT = (
    "A red fox sitting in a sunlit alpine meadow at dawn, dew on the grass, soft golden backlight, snow-capped "
    "mountains in the hazy distance, shallow depth of field, natural wildlife photograph"
)
BAKERY_PROMPT = (
    "A cozy corner bakery storefront in the early morning, a wooden sign above the door reading 'MORNING GLORY "
    "BAKERY' in cream-colored letters, a window full of fresh bread and pastries, warm light inside, wet cobblestone "
    "street, photograph"
)


class TestImageGeneratorFlux3Klein:
    @pytest.mark.slow
    def test_flux3_klein_text_to_image(self):
        ImageGeneratorTestHelper.assert_matches_reference_image(
            reference_image_path="reference_flux3_klein_fox_meadow.png",
            output_image_path="output_flux3_klein_fox_meadow.png",
            model_class=Flux3Klein,
            model_config=ModelConfig.flux3_image_klein(),
            quantize=None,
            steps=16,
            seed=5,
            height=512,
            width=512,
            prompt=FOX_PROMPT,
            mismatch_threshold=0.15,
        )

    @pytest.mark.slow
    def test_flux3_klein_base_text_to_image(self):
        ImageGeneratorTestHelper.assert_matches_reference_image(
            reference_image_path="reference_flux3_klein_base_bakery.png",
            output_image_path="output_flux3_klein_base_bakery.png",
            model_class=Flux3Klein,
            model_config=ModelConfig.flux3_image_klein_base(),
            quantize=None,
            steps=20,
            seed=7,
            height=512,
            width=512,
            guidance=3.5,
            prompt=BAKERY_PROMPT,
            mismatch_threshold=0.15,
        )
