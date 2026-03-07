import pytest

from mflux.models.common.config import ModelConfig
from mflux.models.fibo.variants.edit import FIBOEdit
from tests.image_generation.helpers.image_generation_test_helper import ImageGeneratorTestHelper

FISTBUMP_PROMPT = """
{
  "short_description": "A close-up shot of a Black man's hand making a fistbump gesture towards the camera. He is wearing a plain white t-shirt. The background is a softly blurred indoor setting with a window and curtains.",
  "objects": [
    {
      "description": "A Black man's hand, with visible knuckles and skin texture, making a fistbump gesture.",
      "location": "center foreground",
      "relationship": "The hand is the primary subject, making contact with the camera.",
      "relative_size": "large within frame",
      "shape_and_color": "Human hand shape, dark brown skin tone.",
      "texture": "Smooth skin with visible knuckles.",
      "appearance_details": "Fingers are curled into a fist, thumb is extended.",
      "orientation": "facing forward, fist extended towards the viewer"
    },
    {
      "description": "A Black man's torso and lower face, partially visible, wearing a white t-shirt.",
      "location": "center midground",
      "relationship": "The man is the owner of the hand, providing context for the gesture.",
      "relative_size": "medium",
      "shape_and_color": "Human torso and face shape, dark brown skin tone, white shirt.",
      "texture": "Smooth skin, soft fabric of the t-shirt.",
      "appearance_details": "He has a short beard and mustache. His expression is serious and direct.",
      "pose": "Upper body visible, arm extended forward for a fistbump.",
      "expression": "serious, direct gaze",
      "clothing": "plain white crew-neck t-shirt",
      "action": "fistbumping the camera",
      "gender": "male",
      "skin_tone_and_texture": "dark brown, smooth skin",
      "orientation": "upright, facing forward"
    }
  ],
  "background_setting": "A softly blurred indoor setting, featuring a light gray wall on the left, a window with natural light streaming through on the right, and sheer white curtains partially drawn.",
  "lighting": {
    "conditions": "bright indoor lighting, natural light from a window",
    "direction": "side-lit from right",
    "shadows": "soft shadows are cast on the left side of the hand and face, indicating light from the right window."
  },
  "aesthetics": {
    "composition": "centered, portrait composition with the hand as the focal point",
    "color_scheme": "neutral tones with a pop of white from the shirt and natural light.",
    "mood_atmosphere": "direct, engaging, slightly serious.",
    "photographic_characteristics": {
      "depth_of_field": "shallow",
      "focus": "sharp focus on the hand and face, with a blurred background",
      "camera_angle": "eye-level",
      "lens_focal_length": "standard lens (e.g., 35mm-50mm)"
    },
    "style_medium": "photograph",
    "artistic_style": "realistic, naturalistic",
    "preference_score": "very high",
    "aesthetic_score": "very high"
  },
  "context": "This is a portrait photograph, potentially for a social media profile, a casual greeting, or a promotional image emphasizing a direct and engaging interaction.",
  "edit_instruction": "Make the hand fistbump the camera instead of showing a flat palm."
}
"""


class TestImageGeneratorFiboEdit:
    @pytest.mark.slow
    def test_image_generation_fibo_edit_fistbump(self):
        ImageGeneratorTestHelper.assert_matches_reference_image(
            reference_image_path="reference_fibo_edit_fistbump.png",
            output_image_path="output_fibo_edit_fistbump.png",
            model_class=FIBOEdit,
            model_config=ModelConfig.fibo_edit(),
            quantize=8,
            steps=30,
            seed=1772896237,
            height=384,
            width=640,
            guidance=3.5,
            image_path="reference_upscaled.png",
            prompt=FISTBUMP_PROMPT,
            mismatch_threshold=0.15,
        )
