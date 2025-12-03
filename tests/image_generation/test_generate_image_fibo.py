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
      "shape_and_color": "Round head, large eyes, voluminous body shape. Predominantly brown and gray with silver accents.",
      "texture": "extremely soft, fluffy, downy feathers",
      "appearance_details": "Wide, dark pupils in large, light-colored eyes. Small, delicate beak. Visible ear tufts.",
      "number_of_objects": 1,
      "pose": "Sitting upright, facing forward.",
      "expression": "curious and gentle",
      "action": "looking directly at the viewer",
      "gender": "unknown",
      "orientation": "upright"
    }
  ],
  "background_setting": "A dense, dark forest at night. The trees are silhouetted against a dark sky, with subtle hints of moonlight filtering through the leaves. The overall impression is one of depth and nocturnal serenity.",
  "lighting": {
    "conditions": "night, moonlight",
    "direction": "side-lit from the left, with some ambient light from above",
    "shadows": "soft, elongated shadows cast by the owl and branches, adding depth"
  },
  "aesthetics": {
    "composition": "centered framing, with the owl as the clear focal point",
    "color_scheme": "cool blues and grays of moonlight contrasting with the warm browns and grays of the owl and forest",
    "mood_atmosphere": "magical, serene, whimsical",
    "aesthetic_score": "very high",
    "preference_score": "very high"
  },
  "photographic_characteristics": {
    "depth_of_field": "shallow, with the background softly blurred",
    "focus": "sharp focus on the owl's face and eyes",
    "camera_angle": "eye-level",
    "lens_focal_length": "medium portrait lens"
  },
  "style_medium": "digital illustration",
  "text_render": [],
  "context": "This image is a charming illustration, suitable for a children's book, a whimsical art print, or a decorative piece for a nature-themed space.",
  "artistic_style": "storybook, whimsical, detailed"
}
"""

OWL_PROMPT_REFINED = """
{
  "short_description": "A hyper-detailed, ultra-fluffy owl sitting in the trees at night, looking directly at the camera with wide, adorable, expressive eyes. Its feathers are soft and voluminous, catching the cool moonlight with subtle silver highlights. The owl's gaze is curious and full of charm, giving it a whimsical, storybook-like personality.",
  "objects": [
    {
      "description": "An adorable, fluffy owl with large, expressive eyes and soft, voluminous feathers. Its plumage is a mix of white and subtle silver highlights from the moonlight.",
      "location": "center",
      "relationship": "The owl is the sole subject, perched comfortably within its environment.",
      "relative_size": "large within frame",
      "shape_and_color": "Round head, large eyes, voluminous body shape. Predominantly white with silver accents.",
      "texture": "extremely soft, fluffy, downy feathers",
      "appearance_details": "Wide, dark pupils in large, light-colored eyes. Small, delicate beak. Visible ear tufts.",
      "number_of_objects": 1,
      "pose": "Sitting upright, facing forward.",
      "expression": "curious and gentle",
      "action": "looking directly at the viewer",
      "gender": "unknown",
      "orientation": "upright"
    }
  ],
  "background_setting": "A dense, dark forest at night. The trees are silhouetted against a dark sky, with subtle hints of moonlight filtering through the leaves. The overall impression is one of depth and nocturnal serenity.",
  "lighting": {
    "conditions": "night, moonlight",
    "direction": "side-lit from the left, with some ambient light from above",
    "shadows": "soft, elongated shadows cast by the owl and branches, adding depth"
  },
  "aesthetics": {
    "composition": "centered framing, with the owl as the clear focal point",
    "color_scheme": "cool blues and grays of moonlight contrasting with the white of the owl and forest",
    "mood_atmosphere": "magical, serene, whimsical",
    "aesthetic_score": "very high",
    "preference_score": "very high"
  },
  "photographic_characteristics": {
    "depth_of_field": "shallow, with the background softly blurred",
    "focus": "sharp focus on the owl's face and eyes",
    "camera_angle": "eye-level",
    "lens_focal_length": "medium portrait lens"
  },
  "style_medium": "digital illustration",
  "text_render": [],
  "context": "This image is a charming illustration, suitable for a children's book, a whimsical art print, or a decorative piece for a nature-themed space.",
  "artistic_style": "storybook, whimsical, detailed"
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
