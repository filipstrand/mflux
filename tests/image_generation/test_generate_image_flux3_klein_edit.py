import pytest

from mflux.models.common.config import ModelConfig
from mflux.models.flux3.variants import Flux3KleinEdit
from tests.image_generation.helpers.image_generation_test_helper import ImageGeneratorTestHelper

# Edits work best with a full caption of the result, not an instruction.
NIGHT_PROMPT = (
    "A golden retriever puppy sitting upright on a gray stone patio at night under a full moon, holding a yellow tulip "
    "in its mouth, cool blue moonlight, deep shadows, stars in the sky above dark cypress trees and a wooden lattice "
    "fence, photograph"
)


class TestImageGeneratorFlux3KleinEdit:
    @pytest.mark.slow
    def test_flux3_klein_edit_night(self):
        ImageGeneratorTestHelper.assert_matches_reference_image(
            reference_image_path="reference_flux3_klein_edit_night.png",
            output_image_path="output_flux3_klein_edit_night.png",
            model_class=Flux3KleinEdit,
            model_config=ModelConfig.flux3_image_klein(),
            quantize=None,
            steps=16,
            seed=42,
            height=512,
            width=320,
            image_paths=["unsplash_dog.jpg"],
            prompt=NIGHT_PROMPT,
            mismatch_threshold=0.15,
        )

    @pytest.mark.slow
    def test_flux3_klein_base_edit_night(self):
        ImageGeneratorTestHelper.assert_matches_reference_image(
            reference_image_path="reference_flux3_klein_base_edit_night.png",
            output_image_path="output_flux3_klein_base_edit_night.png",
            model_class=Flux3KleinEdit,
            model_config=ModelConfig.flux3_image_klein_base(),
            quantize=None,
            steps=20,
            seed=42,
            height=512,
            width=320,
            guidance=3.5,
            image_paths=["unsplash_dog.jpg"],
            prompt=NIGHT_PROMPT,
            mismatch_threshold=0.15,
        )
