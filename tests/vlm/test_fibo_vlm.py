import json
from pathlib import Path

import pytest
from PIL import Image

from mflux.models.fibo_vlm.model.fibo_vlm import FiboVLM
from tests.image_generation.test_generate_image_fibo import OWL_PROMPT, OWL_PROMPT_REFINED

INSPIRE_PROMPT = """
{
  "short_description": "A charming, stylized illustration of a young owl sitting on a mossy forest floor. The owl is facing forward with large, expressive eyes and fluffy ear tufts. The background depicts a dark, atmospheric forest with tall trees and dappled moonlight, creating a mystical and serene environment. The overall impression is one of innocence and enchantment, suitable for children's literature or fantasy art.",
  "objects": [
    {
      "description": "A small, fluffy owl with large, round, dark eyes and prominent ear tufts. Its plumage is a soft, mottled grey and white, with subtle patterns suggesting feathers. It has a small, yellow beak.",
      "location": "center",
      "relationship": "The owl is the sole subject, positioned centrally.",
      "relative_size": "medium within frame",
      "shape_and_color": "Round body, large circular eyes, grey and white coloration.",
      "texture": "soft, fluffy feathers",
      "appearance_details": "Its eyes have a glossy sheen, and its ear tufts are particularly bushy.",
      "number_of_objects": 1,
      "pose": "Sitting upright, with its body facing forward and its head slightly tilted.",
      "expression": "curious and innocent",
      "action": "sitting and observing",
      "gender": "neutral",
      "orientation": "upright"
    },
    {
      "description": "A patch of vibrant green moss covering the ground in the foreground and around the owl.",
      "location": "bottom foreground and surrounding the owl's feet",
      "relationship": "The moss serves as the ground on which the owl is perched.",
      "relative_size": "small",
      "shape_and_color": "Irregular patches, vibrant green.",
      "texture": "soft, velvety, organic",
      "appearance_details": "Individual blades of grass and small twigs are visible within the moss.",
      "orientation": "horizontal"
    },
    {
      "description": "Several tall, dark tree trunks forming the background.",
      "location": "background, left, right, and top edges",
      "relationship": "These trees create the forest environment behind the owl.",
      "relative_size": "large",
      "shape_and_color": "Vertical, dark brown and grey.",
      "texture": "rough bark",
      "appearance_details": "Some trunks have patches of moss or lichen. The overall effect is dense and shadowy.",
      "orientation": "vertical"
    }
  ],
  "background_setting": "A dense, mystical forest at night. Tall trees with dark bark dominate the scene, with a hint of moonlight filtering through the canopy. The ground is covered in lush green moss and scattered leaves, with faint glowing orbs suggesting magical elements.",
  "lighting": {
    "conditions": "dim, atmospheric night lighting",
    "direction": "diffused from above and behind the trees",
    "shadows": "soft, elongated shadows cast by the trees, creating depth"
  },
  "aesthetics": {
    "composition": "centered composition with the owl as the focal point",
    "color_scheme": "muted earth tones with vibrant green accents and deep blues/greys in the background",
    "mood_atmosphere": "enchanting, serene, mystical",
    "aesthetic_score": "very high",
    "preference_score": "very high"
  },
  "photographic_characteristics": {
    "depth_of_field": "shallow, with the background softly blurred",
    "focus": "sharp focus on the owl",
    "camera_angle": "eye-level",
    "lens_focal_length": "standard lens (e.g., 50mm)"
  },
  "style_medium": "digital illustration",
  "text_render": [],
  "context": "This image is suitable for a children's book illustration, a fantasy-themed greeting card, or concept art for a magical game.",
  "artistic_style": "stylized, whimsical, detailed"
}
"""

SKYSCRAPERS_INSPIRE_PROMPT = """
{
  "short_description": "A dramatic, low-angle shot of several modern skyscrapers in black and white, emphasizing their towering height and geometric forms against a bright, overcast sky. The buildings feature repetitive window patterns and varying architectural details, creating a sense of urban grandeur and scale.",
  "objects": [
    {
      "description": "A tall skyscraper with a facade of numerous rectangular windows, arranged in a grid pattern. The building has a slightly curved or angled design, giving it a dynamic appearance.",
      "location": "center-right foreground",
      "relationship": "This is the most prominent building, dominating the right side of the frame and serving as a primary focal point.",
      "relative_size": "large within frame",
      "shape_and_color": "Rectangular and angular forms, dark grey to black with bright white window reflections.",
      "texture": "Smooth, reflective glass and concrete, with visible mullions.",
      "appearance_details": "The windows appear to be uniformly spaced, and some sections of the building have protruding architectural elements.",
      "orientation": "vertical"
    },
    {
      "description": "A large, dark, flat roof or top section of a building, extending into the upper-left corner of the frame. It has a slightly textured surface.",
      "location": "top-left foreground",
      "relationship": "It frames the upper-left portion of the image, contrasting with the bright sky and the more distant buildings.",
      "relative_size": "large within frame",
      "shape_and_color": "Flat, angular, dark grey to black.",
      "texture": "Rough, matte concrete or stone.",
      "appearance_details": "The surface shows subtle imperfections and variations in tone.",
      "orientation": "horizontal"
    },
    {
      "description": "A tall, slender skyscraper with a facade composed of many small, rectangular windows. Its design appears more traditional with vertical emphasis.",
      "location": "bottom-right midground",
      "relationship": "It stands behind and to the right of the central skyscraper, adding depth to the urban landscape.",
      "relative_size": "medium",
      "shape_and_color": "Tall, slender rectangular form, dark grey with bright white window reflections.",
      "texture": "Smooth glass and concrete.",
      "appearance_details": "The windows are tightly packed, creating a uniform pattern.",
      "orientation": "vertical"
    },
    {
      "description": "A series of less distinct skyscrapers and urban structures, partially obscured by atmospheric haze or distance.",
      "location": "midground and background",
      "relationship": "These buildings form the distant urban backdrop, providing context for the foreground structures.",
      "relative_size": "small",
      "shape_and_color": "Various rectangular and indistinct shapes, dark grey.",
      "texture": "Indistinct due to distance and haze.",
      "appearance_details": "Their details are softened by the monochromatic atmosphere.",
      "number_of_objects": 5,
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
    "composition": "dynamic and dramatic low-angle composition, with leading lines created by the converging architectural elements drawing the eye upwards.",
    "color_scheme": "monochromatic (black and white) with a wide range of greys, emphasizing form and texture.",
    "mood_atmosphere": "grand, imposing, modern, and slightly mysterious.",
    "aesthetic_score": "very high",
    "preference_score": "very high"
  },
  "photographic_characteristics": {
    "depth_of_field": "deep",
    "focus": "sharp focus on the foreground and midground buildings, with slight softening towards the distant background.",
    "camera_angle": "very low angle, looking upwards from street level.",
    "lens_focal_length": "wide-angle"
  },
  "style_medium": "photograph",
  "text_render": [],
  "context": "An architectural photograph, likely intended for art prints, urban exploration documentation, or a design magazine, focusing on the grandeur of modern cityscapes.",
  "artistic_style": "monochromatic, architectural, dramatic"
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
