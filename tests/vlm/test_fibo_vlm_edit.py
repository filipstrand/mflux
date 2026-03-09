import json
from pathlib import Path

import pytest
from PIL import Image

from mflux.models.fibo_vlm.model.fibo_vlm import FiboVLM

FISTBUMP_EDIT_PROMPT = """
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
      "skin_tone_and_texture": "dark brown skin tone, smooth texture",
      "orientation": "upright, facing forward"
    }
  ],
  "background_setting": "A softly blurred indoor setting, featuring a light gray wall on the left, a window with natural light streaming through on the right, and light-colored curtains partially drawn.",
  "lighting": {
    "conditions": "bright indoor lighting, natural light from a window",
    "direction": "side-lit from right, with some front fill",
    "shadows": "soft, subtle shadows on the left side of the hand and arm, indicating light from the right"
  },
  "aesthetics": {
    "composition": "centered, portrait composition with a strong foreground element",
    "color_scheme": "neutral tones (white, gray, brown) with natural light accents",
    "mood_atmosphere": "direct, engaging, slightly serious",
    "photographic_characteristics": {
      "depth_of_field": "shallow, with the hand in sharp focus and the background blurred",
      "focus": "sharp focus on the hand and the man's face",
      "camera_angle": "eye-level",
      "lens_focal_length": "standard lens (e.g., 35mm-50mm) or portrait lens (e.g., 50mm-85mm)"
    },
    "style_medium": "photograph",
    "artistic_style": "realistic, naturalistic, direct",
    "preference_score": "very high",
    "aesthetic_score": "very high"
  },
  "context": "This is a portrait photograph, potentially for a social media profile, a casual editorial piece, or a personal statement.",
  "edit_instruction": "Make the hand fistbump the camera instead of showing a flat palm."
}
"""


@pytest.fixture
def vlm():
    return FiboVLM(model_id="briaai/FIBO-vlm", quantize=8)


@pytest.mark.slow
def test_vlm_edit_json_from_image(vlm):
    # given
    resource_dir = Path(__file__).parent.parent / "resources"
    image_path = resource_dir / "reference_upscaled.png"
    image = Image.open(image_path).convert("RGB")

    # when
    json_output = vlm.edit(
        image=image,
        edit_instruction="Make the hand fistbump the camera instead of showing a flat palm",
        seed=42,
    )

    # then
    assert json_output == FISTBUMP_EDIT_PROMPT, "Generated JSON output does not match expected output exactly."
    assert isinstance(json.loads(json_output), dict), "Output should be a valid JSON object"


def test_build_edit_messages_uses_mask_specific_prompt():
    image = Image.new("RGB", (8, 8), (255, 255, 255))

    messages = FiboVLM._build_messages(
        task="edit",
        image=image,
        editing_instructions="add a tree",
        use_mask=True,
    )

    assert "Image Completion" in messages[0]["content"]
    assert "grey areas as regions to be filled or extended" in messages[1]["content"][1]["text"]


def test_edit_fallback_injects_original_instruction_when_missing(monkeypatch):
    calls = []

    def fake_generate_internal(self, task, **kwargs):
        calls.append(task)
        if task == "edit":
            raise ValueError("VLM edit output must contain exactly one JSON object.")
        return json.dumps(
            {
                "short_description": "white cat",
                "aesthetics": {"composition": "centered"},
            }
        )

    monkeypatch.setattr(FiboVLM, "_generate_internal", fake_generate_internal)
    vlm = object.__new__(FiboVLM)

    result = json.loads(
        vlm.edit(
            image=Image.new("RGB", (8, 8), (255, 255, 255)),
            edit_instruction="turn the cat color to white",
            seed=42,
        )
    )

    assert calls == ["edit", "refine"]
    assert result["edit_instruction"] == "Turn the cat color to white"
    assert result["aesthetics"]["aesthetic_score"] == "very high"
    assert result["aesthetics"]["preference_score"] == "very high"


def test_edit_with_mask_fails_hard_instead_of_refine_fallback(monkeypatch):
    calls = []

    def fake_generate_internal(self, task, **kwargs):
        calls.append(task)
        raise ValueError("VLM edit output must contain exactly one JSON object.")

    monkeypatch.setattr(FiboVLM, "_generate_internal", fake_generate_internal)
    vlm = object.__new__(FiboVLM)

    with pytest.raises(ValueError, match="refusing fallback"):
        vlm.edit(
            image=Image.new("RGB", (8, 8), (255, 255, 255)),
            edit_instruction="fill the masked area with grass",
            use_mask=True,
            seed=42,
        )

    assert calls == ["edit"]
