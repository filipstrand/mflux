import json
from pathlib import Path

import pytest
from PIL import Image

from mflux.models.fibo_vlm.model.fibo_vlm import FiboVLM
from tests.image_generation.test_generate_image_fibo import OWL_PROMPT, OWL_PROMPT_REFINED

INSPIRE_PROMPT = """
{
  "short_description": "A surreal and dreamlike scene unfolds in a vast, empty landscape under a starry night sky. In the foreground, a lone, weathered wooden structure resembling a small house or shed stands, with a single window emitting a warm, yellow light. To the left, a cluster of similarly constructed, smaller wooden forms are scattered across the ground. The overall atmosphere is one of quiet solitude and mystery, with a touch of cosmic wonder.",
  "objects": [
    {
      "description": "A solitary, rustic wooden structure with a pitched roof, resembling a small cabin or shed. It has a single window that glows with a warm, inviting yellow light.",
      "location": "center-right foreground",
      "relationship": "It is the most prominent object in the scene, standing alone in the landscape.",
      "relative_size": "medium",
      "shape_and_color": "Rectangular base with a triangular roof, made of weathered, light brown wood. The window is square and emits yellow light.",
      "texture": "rough, weathered wood",
      "appearance_details": "The wood appears aged and slightly worn. The window light casts a soft glow.",
      "orientation": "upright"
    },
    {
      "description": "A cluster of smaller, similarly styled wooden structures, appearing as fragments or derelict sheds. They are scattered across the ground to the left of the main structure.",
      "location": "left midground",
      "relationship": "These structures appear to be remnants or companions to the main wooden structure, creating a sense of a lost settlement.",
      "relative_size": "small",
      "shape_and_color": "Various rectangular and cuboid shapes, light brown wood.",
      "texture": "rough, weathered wood",
      "appearance_details": "Some have visible doorways or openings. They are less detailed than the main structure.",
      "number_of_objects": 5,
      "orientation": "various, mostly upright"
    },
    {
      "description": "A vast, star-filled night sky that transitions into a deep, dark expanse.",
      "location": "top half of the image",
      "relationship": "It forms the backdrop for the entire scene, emphasizing the cosmic scale.",
      "relative_size": "large",
      "shape_and_color": "Vast, dark blue to black gradient, filled with countless white and yellow stars.",
      "texture": "smooth, celestial",
      "appearance_details": "The stars are sharp points of light, some forming constellations. A subtle nebula-like glow is visible in the upper right.",
      "orientation": "horizontal"
    },
    {
      "description": "The ground is a flat, dark, and somewhat textured surface, resembling a desolate plain or a cosmic dust field.",
      "location": "bottom half of the image",
      "relationship": "It serves as the base for the wooden structures and the overall environment.",
      "relative_size": "large",
      "shape_and_color": "Dark, muted brown or charcoal grey, with subtle variations in tone.",
      "texture": "slightly granular, dusty",
      "appearance_details": "Appears to be a fine, powdery substance. Some faint indentations are visible.",
      "orientation": "horizontal"
    }
  ],
  "background_setting": "An expansive, desolate landscape under a clear, star-filled night sky. The ground is a flat, dark plain. The horizon is indistinct, blending into the sky.",
  "lighting": {
    "conditions": "night, starlight, internal glow",
    "direction": "ambient from stars, warm light emanating from the window",
    "shadows": "soft, elongated shadows cast by the structures, suggesting a low light source"
  },
  "aesthetics": {
    "composition": "centered framing with the main structure slightly off-center, creating a sense of balance and depth",
    "color_scheme": "limited palette of dark blues, browns, and yellows, with high contrast between the starry sky and the earthly structures",
    "mood_atmosphere": "surreal, mysterious, serene, cosmic",
    "aesthetic_score": "very high",
    "preference_score": "very high"
  },
  "photographic_characteristics": {
    "depth_of_field": "deep",
    "focus": "sharp focus on the wooden structures and stars",
    "camera_angle": "eye-level",
    "lens_focal_length": "standard lens"
  },
  "style_medium": "digital art",
  "text_render": [],
  "context": "This image evokes a sense of wonder and isolation, suitable for concept art for a science fiction or fantasy game, or as a piece of atmospheric digital art.",
  "artistic_style": "surreal, cosmic, minimalist"
}
"""


@pytest.fixture
def vlm():
    return FiboVLM(model_id="briaai/FIBO-vlm")


def test_vlm_generate_json(vlm):
    # given
    input_prompt = "A hyper-detailed, ultra-fluffy owl sitting in the trees at night, looking directly at the camera with wide, adorable, expressive eyes. Its feathers are soft and voluminous, catching the cool moonlight with subtle silver highlights. The owl's gaze is curious and full of charm, giving it a whimsical, storybook-like personality."

    # when
    json_output = vlm.generate(
        top_p=0.9,
        temperature=0.2,
        max_tokens=4096,
        stop=["<|im_end|>", "<|end_of_text|>"],
        task="generate",
        prompt=input_prompt,
        seed=42,
    )

    # then
    assert json_output == OWL_PROMPT, "Generated JSON output does not match expected output exactly."
    assert isinstance(json.loads(json_output), dict), "Output should be a valid JSON object"


def test_vlm_refine_json_text_only(vlm):
    # given/when
    json_output = vlm.generate(
        top_p=0.9,
        temperature=0.2,
        max_tokens=4096,
        stop=["<|im_end|>", "<|end_of_text|>"],
        task="refine",
        structured_prompt=OWL_PROMPT,
        editing_instructions="make the owl white color but keep everything else exactly the same",
        seed=42,
    )

    # then
    assert json_output == OWL_PROMPT_REFINED, "Generated JSON output does not match expected output exactly."
    assert isinstance(json.loads(json_output), dict), "Output should be a valid JSON object"


def test_vlm_inspire_json_from_image(vlm):
    # given
    resource_dir = Path(__file__).parent.parent / "resources"
    image_path = resource_dir / "reference_fibo.png"
    image = Image.open(image_path).convert("RGB")

    # when
    json_output = vlm.generate(
        top_p=0.9,
        temperature=0.2,
        max_tokens=4096,
        stop=["<|im_end|>", "<|end_of_text|>"],
        task="inspire",
        image=image,
        seed=42,
    )

    # then
    assert json_output == INSPIRE_PROMPT, "Generated JSON output does not match expected output exactly."
    assert isinstance(json.loads(json_output), dict), "Output should be a valid JSON object"
