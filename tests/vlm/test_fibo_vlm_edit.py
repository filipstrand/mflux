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
      "texture": "Smooth skin with visible lines and creases.",
      "appearance_details": "Fingers are curled into a fist, thumb is extended to meet the camera's implied surface.",
      "orientation": "facing forward, fist extended towards the viewer"
    },
    {
      "description": "A Black man's upper torso and neck, partially visible, wearing a white t-shirt.",
      "location": "center midground",
      "relationship": "The hand belongs to this person.",
      "relative_size": "medium",
      "shape_and_color": "Human upper body shape, dark brown skin tone, white shirt.",
      "texture": "Smooth skin, soft cotton fabric.",
      "appearance_details": "Only the neck and upper chest are visible above the hand.",
      "pose": "Upper body is slightly angled, head is not fully visible.",
      "expression": "Not visible, but implied serious or direct gaze.",
      "clothing": "A plain white crew-neck t-shirt.",
      "action": "Making a fistbump.",
      "gender": "male",
      "skin_tone_and_texture": "Dark brown skin tone, smooth texture.",
      "orientation": "vertical"
    }
  ],
  "background_setting": "A softly blurred indoor setting, featuring a light gray wall on the left, a window with bright, diffused light on the right, and a light-colored curtain partially visible on the far right.",
  "lighting": {
    "conditions": "bright indoor lighting",
    "direction": "front-lit with some backlighting from the window",
    "shadows": "soft, subtle shadows on the hand and arm, indicating diffused light"
  },
  "aesthetics": {
    "composition": "centered, close-up shot focusing on the hand gesture",
    "color_scheme": "neutral tones with a pop of white from the shirt and warm skin tones",
    "mood_atmosphere": "direct, engaging, slightly serious",
    "photographic_characteristics": {
      "depth_of_field": "shallow",
      "focus": "sharp focus on the hand, with the background softly blurred",
      "camera_angle": "eye-level",
      "lens_focal_length": "standard lens (e.g., 35mm-50mm)"
    },
    "style_medium": "photograph",
    "artistic_style": "realistic, naturalistic",
    "preference_score": "very high",
    "aesthetic_score": "very high"
  },
  "context": "This is a portrait photograph, likely intended for social media, a personal profile, or a casual editorial feature, emphasizing a direct and engaging interaction with the viewer.",
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
