import json
from pathlib import Path

import pytest
from PIL import Image

from mflux.models.fibo_vlm.model.fibo_vlm import FiboVLM
from tests.image_generation.test_generate_image_fibo import OWL_PROMPT, OWL_PROMPT_REFINED

INSPIRE_PROMPT = """
{
  "short_description": "A charming, stylized illustration of a young owl sitting on a mossy forest floor. The owl is facing forward with large, expressive eyes and fluffy ear tufts. It is surrounded by a softly blurred forest environment with tall trees and dappled moonlight. The overall impression is one of innocence and nocturnal wonder, rendered in a gentle, painterly style.",
  "objects": [
    {
      "description": "A young, anthropomorphic owl with large, round eyes and prominent ear tufts. Its plumage is a soft, mottled grey and beige, with distinct feather patterns.",
      "location": "center",
      "relationship": "The owl is the sole prominent subject, resting on the ground.",
      "relative_size": "medium-to-large within frame",
      "shape_and_color": "Round body, large circular eyes, conical beak. Predominantly grey and beige.",
      "texture": "soft, feathery",
      "appearance_details": "Large, dark pupils in its eyes, a small yellow beak, and delicate talons.",
      "number_of_objects": 1,
      "pose": "Sitting upright, facing forward.",
      "expression": "curious and gentle",
      "action": "resting",
      "gender": "neutral",
      "skin_tone_and_texture": "N/A (owl)",
      "orientation": "upright"
    },
    {
      "description": "A patch of vibrant green moss covering the ground in the foreground and around the owl.",
      "location": "bottom foreground and midground",
      "relationship": "The owl is perched on this mossy ground.",
      "relative_size": "medium",
      "shape_and_color": "Irregular, organic shapes, vibrant green.",
      "texture": "soft, velvety, slightly uneven",
      "appearance_details": "Appears lush and damp, with small variations in shade.",
      "orientation": "horizontal"
    },
    {
      "description": "Several tall, slender tree trunks forming the background.",
      "location": "background",
      "relationship": "They create the forest environment behind the owl.",
      "relative_size": "large",
      "shape_and_color": "Vertical, cylindrical shapes. Dark brown and grey.",
      "texture": "rough bark",
      "appearance_details": "Some trunks are covered in moss. They are out of focus.",
      "orientation": "vertical"
    },
    {
      "description": "A few delicate, glowing orbs scattered in the background, resembling fireflies or distant lights.",
      "location": "background, scattered",
      "relationship": "They add a magical element to the forest scene.",
      "relative_size": "small",
      "shape_and_color": "Small, spherical, glowing yellow-white.",
      "texture": "N/A",
      "appearance_details": "Softly blurred, suggesting distance.",
      "orientation": "N/A"
    }
  ],
  "background_setting": "A mystical forest at night, with tall, ancient trees, a soft undergrowth of moss, and a hint of moonlight filtering through the canopy. The atmosphere is serene and enchanting.",
  "lighting": {
    "conditions": "soft, ambient moonlight",
    "direction": "diffused from above and behind",
    "shadows": "soft, elongated shadows cast by the trees, minimal on the owl"
  },
  "aesthetics": {
    "composition": "centered framing with the owl as the main focal point",
    "color_scheme": "cool, muted tones of blues, greys, and greens, with warm accents from the owl's eyes and beak",
    "mood_atmosphere": "magical, serene, innocent",
    "aesthetic_score": "very high",
    "preference_score": "very high"
  },
  "photographic_characteristics": {
    "depth_of_field": "shallow, with a strong bokeh effect in the background",
    "focus": "sharp focus on the owl",
    "camera_angle": "eye-level",
    "lens_focal_length": "standard lens (e.g., 50mm)"
  },
  "style_medium": "digital illustration",
  "text_render": [],
  "context": "This image is suitable for children's book illustrations, fantasy-themed art, or decorative prints.",
  "artistic_style": "stylized, painterly, whimsical"
}
"""

SKYSCRAPERS_INSPIRE_PROMPT = """
{
  "short_description": "A dramatic, low-angle shot of several modern skyscrapers in black and white, emphasizing their towering height and geometric forms against a bright, overcast sky. The buildings feature repetitive window patterns and varying architectural details, creating a sense of urban grandeur and scale.",
  "objects": [
    {
      "description": "A tall skyscraper with a facade of numerous rectangular windows, arranged in a grid pattern. The building has a slightly curved or angled top section.",
      "location": "center-right foreground",
      "relationship": "It is the most prominent building, dominating the right side of the frame and appearing to recede into the background.",
      "relative_size": "large within frame",
      "shape_and_color": "Rectangular with a complex, multi-faceted top; shades of gray and black.",
      "texture": "Smooth concrete and glass, with visible window frames.",
      "appearance_details": "The windows reflect the bright sky, appearing light gray or white. Some sections of the facade have decorative elements or protrusions.",
      "orientation": "vertical"
    },
    {
      "description": "A long, rectangular skyscraper with a facade of regularly spaced, horizontal windows.",
      "location": "left midground",
      "relationship": "It stands to the left of the central skyscraper, appearing slightly further back and narrower.",
      "relative_size": "medium",
      "shape_and_color": "Tall, slender rectangle; shades of gray and black.",
      "texture": "Smooth concrete and glass.",
      "appearance_details": "The horizontal lines of windows create a strong sense of rhythm. Its top is flat and extends horizontally.",
      "orientation": "vertical"
    },
    {
      "description": "A skyscraper with a facade featuring larger, rectangular windows that are not perfectly uniform, giving a slightly more textured appearance.",
      "location": "center-left midground",
      "relationship": "It is positioned behind the left midground skyscraper and to the left of the central skyscraper, partially obscured by the foreground elements.",
      "relative_size": "medium",
      "shape_and_color": "Tall, slender rectangle; shades of gray and black.",
      "texture": "Smooth concrete and glass, with subtle variations in window size.",
      "appearance_details": "Its upper section has a distinct, stepped design.",
      "orientation": "vertical"
    },
    {
      "description": "A building with a facade characterized by smaller, square-like windows, creating a more intricate pattern.",
      "location": "bottom-right foreground",
      "relationship": "It is partially visible at the bottom right, appearing to be closer to the viewer than the central skyscraper.",
      "relative_size": "small",
      "shape_and_color": "Rectangular base with a repeating pattern of small squares; shades of gray and black.",
      "texture": "Smooth concrete and glass.",
      "appearance_details": "The windows are arranged in a dense, grid-like pattern, almost appearing pixelated.",
      "orientation": "vertical"
    }
  ],
  "background_setting": "A bright, uniformly white sky, indicative of an overcast day or a foggy atmosphere, which provides a stark contrast to the dark buildings.",
  "lighting": {
    "conditions": "overcast daylight",
    "direction": "diffused from above",
    "shadows": "soft, subtle shadows on the buildings, emphasizing their forms without harsh lines"
  },
  "aesthetics": {
    "composition": "dynamic low-angle composition, with leading lines from the buildings converging towards the top-center, creating a sense of immense height and perspective.",
    "color_scheme": "monochromatic (black and white) with a wide range of grays, emphasizing texture and form.",
    "mood_atmosphere": "grand, imposing, architectural, and slightly mysterious.",
    "aesthetic_score": "very high",
    "preference_score": "very high"
  },
  "photographic_characteristics": {
    "depth_of_field": "deep",
    "focus": "sharp focus on all visible skyscrapers, maintaining clarity across the architectural details.",
    "camera_angle": "very low angle, looking upwards from ground level.",
    "lens_focal_length": "wide-angle lens, to capture the full height and breadth of the buildings."
  },
  "style_medium": "photograph",
  "text_render": [],
  "context": "An architectural photograph, likely intended for art prints, urban exploration blogs, or promotional material for city tourism, focusing on the abstract beauty of modern cityscapes.",
  "artistic_style": "minimalist, abstract, architectural"
}
"""


@pytest.fixture
def vlm():
    return FiboVLM(model_id="briaai/FIBO-vlm", quantize=8)


@pytest.mark.slow
def test_vlm_generate_json(vlm):
    # given
    input_prompt = "A hyper-detailed, ultra-fluffy owl sitting in the trees at night, looking directly at the camera with wide, adorable, expressive eyes. Its feathers are soft and voluminous, catching the cool moonlight with subtle silver highlights. The owl's gaze is curious and full of charm, giving it a whimsical, storybook-like personality."

    # when
    json_output = vlm.generate(prompt=input_prompt, seed=42)

    # then
    assert json_output == OWL_PROMPT, "Generated JSON output does not match expected output exactly."
    assert isinstance(json.loads(json_output), dict), "Output should be a valid JSON object"


@pytest.mark.slow
def test_vlm_refine_json_text_only(vlm):
    # given
    editing_instructions = "make the owl white color but keep everything else exactly the same"

    # when
    json_output = vlm.refine(
        structured_prompt=OWL_PROMPT,
        editing_instructions=editing_instructions,
        seed=42,
    )

    # then
    assert json_output == OWL_PROMPT_REFINED, "Generated JSON output does not match expected output exactly."
    assert isinstance(json.loads(json_output), dict), "Output should be a valid JSON object"


@pytest.mark.slow
def test_vlm_inspire_json_from_image(vlm):
    # given
    resource_dir = Path(__file__).parent.parent / "resources"
    image_path = resource_dir / "reference_fibo.png"
    image = Image.open(image_path).convert("RGB")

    # when - test without prompt to verify image processing works correctly
    json_output = vlm.inspire(image=image, prompt=None, seed=42)

    # then
    assert json_output == INSPIRE_PROMPT, "Generated JSON output does not match expected output exactly."
    assert isinstance(json.loads(json_output), dict), "Output should be a valid JSON object"


@pytest.mark.slow
def test_vlm_inspire_json_from_image_with_prompt(vlm):
    # given
    resource_dir = Path(__file__).parent.parent / "resources"
    image_path = resource_dir / "skyscrapers.jpg"
    image = Image.open(image_path).convert("RGB")
    prompt = "an image of skyscrapers"

    # when - test with prompt to verify prompt-guided image processing works correctly
    json_output = vlm.inspire(image=image, prompt=prompt, seed=42)

    # then
    assert json_output == SKYSCRAPERS_INSPIRE_PROMPT, "Generated JSON output does not match expected output exactly."
    assert isinstance(json.loads(json_output), dict), "Output should be a valid JSON object"
