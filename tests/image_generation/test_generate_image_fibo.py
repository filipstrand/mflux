import pytest

from tests.image_generation.helpers.image_generation_fibo_test_helper import ImageGeneratorFiboTestHelper

OWL_PROMPT = """
{
  "short_description": "A hyper-detailed, ultra-fluffy owl sitting in the trees at night, looking directly at the camera with wide, adorable, expressive eyes. Its feathers are soft and voluminous, catching the cool moonlight with subtle silver highlights. The owl's gaze is curious and full of charm, giving it a whimsical, storybook-like personality.",
  "objects": [
    {
      "description": "An adorable, fluffy owl with large, expressive eyes and soft, voluminous feathers. Its plumage is a mix of earthy tones with subtle silver highlights from the moonlight.",
      "location": "center",
      "relationship": "The owl is the sole subject, perched on a branch.",
      "relative_size": "large within frame",
      "shape_and_color": "Round head, large eyes, rounded body shape. Predominantly shades of brown, grey, and white, with silver highlights.",
      "texture": "extremely soft, fluffy, downy feathers",
      "appearance_details": "Wide, dark pupils in large, light-colored irises. Small, delicate beak. Visible ear tufts.",
      "number_of_objects": 1,
      "pose": "Sitting upright, facing forward.",
      "expression": "curious and gentle",
      "action": "looking directly at the viewer",
      "gender": "unknown",
      "orientation": "upright"
    }
  ],
  "background_setting": "A dark, nocturnal forest environment. Silhouetted trees and branches are visible, with a hint of moonlight filtering through. The overall atmosphere is mysterious and serene.",
  "lighting": {
    "conditions": "moonlight",
    "direction": "side-lit from the left, with some ambient light from above",
    "shadows": "soft, elongated shadows cast by the owl and branches, adding depth"
  },
  "aesthetics": {
    "composition": "centered framing, with the owl filling a significant portion of the frame",
    "color_scheme": "cool color palette dominated by blues, greys, and deep greens, with warm accents from the owl's eyes and subtle browns",
    "mood_atmosphere": "magical, serene, whimsical",
    "aesthetic_score": "very high",
    "preference_score": "very high"
  },
  "photographic_characteristics": {
    "depth_of_field": "shallow, with a softly blurred background",
    "focus": "sharp focus on the owl's face and eyes",
    "camera_angle": "eye-level",
    "lens_focal_length": "medium portrait lens"
  },
  "style_medium": "digital illustration",
  "text_render": [],
  "context": "This image is a whimsical illustration, suitable for a children's book, a fantasy game, or a charming piece of digital art.",
  "artistic_style": "storybook, fantasy, detailed"
}
"""

OWL_PROMPT_REFINED = """
{
  "short_description": "A hyper-detailed, ultra-fluffy owl sitting in the trees at night, looking directly at the camera with wide, adorable, expressive eyes. Its feathers are soft and voluminous, catching the cool moonlight with subtle silver highlights. The owl's gaze is curious and full of charm, giving it a whimsical, storybook-like personality.",
  "objects": [
    {
      "description": "An adorable, fluffy owl with large, expressive eyes and soft, voluminous feathers. Its plumage is a mix of earthy tones with subtle silver highlights from the moonlight.",
      "location": "center",
      "relationship": "The owl is the sole subject, perched on a branch.",
      "relative_size": "large within frame",
      "shape_and_color": "Round head, large eyes, rounded body shape. Predominantly shades of white, with silver highlights.",
      "texture": "extremely soft, fluffy, downy feathers",
      "appearance_details": "Wide, dark pupils in large, light-colored irises. Small, delicate beak. Visible ear tufts.",
      "number_of_objects": 1,
      "pose": "Sitting upright, facing forward.",
      "expression": "curious and gentle",
      "action": "looking directly at the viewer",
      "gender": "unknown",
      "orientation": "upright"
    }
  ],
  "background_setting": "A dark, nocturnal forest environment. Silhouetted trees and branches are visible, with a hint of moonlight filtering through. The overall atmosphere is mysterious and serene.",
  "lighting": {
    "conditions": "moonlight",
    "direction": "side-lit from the left, with some ambient light from above",
    "shadows": "soft, elongated shadows cast by the owl and branches, adding depth"
  },
  "aesthetics": {
    "composition": "centered framing, with the owl filling a significant portion of the frame",
    "color_scheme": "cool color palette dominated by blues, greys, and deep greens, with warm accents from the owl's eyes and subtle browns",
    "mood_atmosphere": "magical, serene, whimsical",
    "aesthetic_score": "very high",
    "preference_score": "very high"
  },
  "photographic_characteristics": {
    "depth_of_field": "shallow, with a softly blurred background",
    "focus": "sharp focus on the owl's face and eyes",
    "camera_angle": "eye-level",
    "lens_focal_length": "medium portrait lens"
  },
  "style_medium": "digital illustration",
  "text_render": [],
  "context": "This image is a whimsical illustration, suitable for a children's book, a fantasy game, or a charming piece of digital art.",
  "artistic_style": "storybook, fantasy, detailed"
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
