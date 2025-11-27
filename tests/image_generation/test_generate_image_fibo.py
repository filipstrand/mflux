import pytest

from tests.image_generation.helpers.image_generation_fibo_test_helper import ImageGeneratorFiboTestHelper

OWL_PROMPT = """
{
  "short_description": "A hyper-detailed, ultra-fluffy owl sitting in the trees at night, looking directly at the camera with wide, adorable, expressive eyes. Its feathers are soft and voluminous, catching the cool moonlight with subtle silver highlights. The owl's gaze is curious and full of charm, giving it a whimsical, storybook-like personality.",
  "objects": [
    {
      "description": "An adorable, fluffy owl with large, expressive eyes and soft, voluminous feathers. Its plumage is a mix of warm browns, grays, and subtle silver highlights from the moonlight.",
      "location": "center",
      "relationship": "The owl is the sole subject, perched comfortably within its environment.",
      "relative_size": "large within frame",
      "shape_and_color": "Round head, large eyes, bulky body, predominantly brown and grey with silver accents.",
      "texture": "Extremely soft, fluffy, and detailed feathers, giving a plush toy-like appearance.",
      "appearance_details": "The eyes are wide, round, and have a glossy, reflective quality, suggesting innocence and curiosity. The beak is small and light-colored, almost blending with the feathers.",
      "orientation": "upright, facing forward"
    }
  ],
  "background_setting": "A dark, blurred forest at night, with hints of tree trunks and foliage visible in the background. The darkness emphasizes the owl as the central focus.",
  "lighting": {
    "conditions": "moonlight",
    "direction": "top-left, casting subtle highlights",
    "shadows": "soft, diffused shadows, contributing to the depth without harshness"
  },
  "aesthetics": {
    "composition": "centered, portrait composition",
    "color_scheme": "cool blues and grays of the night sky and warm browns and grays of the owl, with silver highlights",
    "mood_atmosphere": "whimsical, serene, enchanting, and slightly mysterious",
    "aesthetic_score": "very high",
    "preference_score": "very high"
  },
  "photographic_characteristics": {
    "depth_of_field": "shallow",
    "focus": "sharp focus on the owl's face and eyes",
    "camera_angle": "eye-level",
    "lens_focal_length": "portrait lens (e.g., 50mm-85mm)"
  },
  "style_medium": "digital illustration",
  "text_render": [],
  "context": "A whimsical character illustration, suitable for children's books, animated features, or as a charming decorative art piece.",
  "artistic_style": "hyperrealistic, cute, illustrative"
}
"""

OWL_PROMPT_REFINED = """
{
  "short_description": "A hyper-detailed, ultra-fluffy owl sitting in the trees at night, looking directly at the camera with wide, adorable, expressive eyes. Its feathers are soft and voluminous, catching the cool moonlight with subtle silver highlights. The owl's gaze is curious and full of charm, giving it a whimsical, storybook-like personality.",
  "objects": [
    {
      "description": "An adorable, fluffy owl with large, expressive eyes and soft, voluminous feathers. Its plumage is white with subtle silver highlights from the moonlight.",
      "location": "center",
      "relationship": "The owl is the sole subject, perched comfortably within its environment.",
      "relative_size": "large within frame",
      "shape_and_color": "Round head, large eyes, bulky body, predominantly white with silver accents.",
      "texture": "Extremely soft, fluffy, and detailed feathers, giving a plush toy-like appearance.",
      "appearance_details": "The eyes are wide, round, and have a glossy, reflective quality, suggesting innocence and curiosity. The beak is small and light-colored, almost blending with the feathers.",
      "orientation": "upright, facing forward"
    }
  ],
  "background_setting": "A dark, blurred forest at night, with hints of tree trunks and foliage visible in the background. The darkness emphasizes the owl as the central focus.",
  "lighting": {
    "conditions": "moonlight",
    "direction": "top-left, casting subtle highlights",
    "shadows": "soft, diffused shadows, contributing to the depth without harshness"
  },
  "aesthetics": {
    "composition": "centered, portrait composition",
    "color_scheme": "cool blues and grays of the night sky and white of the owl, with silver accents",
    "mood_atmosphere": "whimsical, serene, enchanting, and slightly mysterious",
    "aesthetic_score": "very high",
    "preference_score": "very high"
  },
  "photographic_characteristics": {
    "depth_of_field": "shallow",
    "focus": "sharp focus on the owl's face and eyes",
    "camera_angle": "eye-level",
    "lens_focal_length": "portrait lens (e.g., 50mm-85mm)"
  },
  "style_medium": "digital illustration",
  "text_render": [],
  "context": "A whimsical character illustration, suitable for children's books, animated features, or as a charming decorative art piece.",
  "artistic_style": "hyperrealistic, cute, illustrative"
}
"""


class TestImageGeneratorFibo:
    @pytest.mark.slow
    def test_image_generation_fibo(self):
        ImageGeneratorFiboTestHelper.assert_matches_reference_image(
            reference_image_path="reference_fibo.png",
            output_image_path="output_fibo.png",
            prompt=OWL_PROMPT,  # Assume this has been generated by the VLM, actual VLM tests are separate
            steps=20,
            seed=42,
            height=352,
            width=640,
            guidance=4.0,
            quantize=8,
        )

    @pytest.mark.slow
    def test_image_generation_fibo_refined_white_owl(self):
        ImageGeneratorFiboTestHelper.assert_matches_reference_image(
            reference_image_path="reference_fibo_white_owl.png",
            output_image_path="output_fibo_white_owl.png",
            prompt=OWL_PROMPT_REFINED,  # Assume this has been refined by the VLM, actual VLM tests are separate
            steps=20,
            seed=42,
            height=352,
            width=640,
            guidance=4.0,
            quantize=8,
        )
