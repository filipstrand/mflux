import json
from pathlib import Path

import pytest
from PIL import Image

from mflux.models.fibo_vlm.model.fibo_vlm import FiboVLM
from tests.image_generation.test_generate_image_fibo import OWL_PROMPT, OWL_PROMPT_REFINED

INSPIRE_PROMPT = """
{
  "short_description": "A charming, stylized illustration of a young owl perched on a mossy tree stump in a dimly lit forest. The owl has large, expressive eyes and soft, feathered textures. The background is a blur of dark trees and foliage, creating a serene and mysterious atmosphere. The overall impression is one of innocence and nature's quiet beauty.",
  "objects": [
    {
      "description": "A young owl with large, round, golden eyes and a small yellow beak. Its plumage is a mix of soft browns, creams, and subtle blues, with distinct feather patterns. It has prominent ear tufts and a fluffy chest.",
      "location": "center",
      "relationship": "perched on a tree stump",
      "relative_size": "medium within frame",
      "shape_and_color": "Rounded body shape, predominantly brown and cream with blueish undertones. Large, round eyes with dark pupils.",
      "texture": "soft, feathery, detailed",
      "appearance_details": "Its eyes have a glossy sheen and appear to be looking directly at the viewer. The feathers have a layered, textured look.",
      "number_of_objects": 1,
      "pose": "Sitting upright, with its body facing forward and its head slightly tilted.",
      "expression": "curious and gentle",
      "action": "perching",
      "gender": "unknown",
      "orientation": "upright"
    },
    {
      "description": "A weathered tree stump covered in lush green moss. It provides a natural perch for the owl.",
      "location": "bottom-center foreground",
      "relationship": "supports the owl",
      "relative_size": "medium",
      "shape_and_color": "Irregular, rounded shape, dark brown wood with vibrant green moss.",
      "texture": "rough wood, soft mossy texture",
      "appearance_details": "The moss is thick and detailed, with small blades of grass growing around the base.",
      "number_of_objects": 1,
      "orientation": "lying on its side"
    },
    {
      "description": "Several slender tree trunks and branches, rendered in dark, muted tones.",
      "location": "background and midground",
      "relationship": "forms the forest environment",
      "relative_size": "large, forming the environment",
      "shape_and_color": "Vertical, elongated shapes in shades of dark brown and grey.",
      "texture": "smooth, bark-like texture",
      "appearance_details": "Some branches have sparse green leaves. The trees are out of focus, creating depth.",
      "orientation": "vertical"
    }
  ],
  "background_setting": "A dense, dark forest with tall trees and undergrowth. The atmosphere is slightly misty or foggy, softening the distant elements.",
  "lighting": {
    "conditions": "dim, atmospheric lighting",
    "direction": "soft, diffused light from the front-left",
    "shadows": "soft, subtle shadows that enhance the depth and mood"
  },
  "aesthetics": {
    "composition": "centered composition with the owl as the primary focal point",
    "color_scheme": "muted earth tones with pops of green and soft blues/greys",
    "mood_atmosphere": "serene, mysterious, innocent",
    "aesthetic_score": "high",
    "preference_score": "high"
  },
  "photographic_characteristics": {
    "depth_of_field": "shallow, with a blurred background",
    "focus": "sharp focus on the owl",
    "camera_angle": "eye-level",
    "lens_focal_length": "standard lens (e.g., 50mm)"
  },
  "style_medium": "digital illustration",
  "text_render": [],
  "context": "This image is suitable for children's book illustrations, nature-themed art, or decorative prints.",
  "artistic_style": "stylized, painterly, cute"
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
      "texture": "Smooth concrete or glass, with visible mullions creating a textured grid.",
      "appearance_details": "The windows reflect the bright sky, appearing as glowing rectangles. The building's structure is robust and imposing.",
      "orientation": "vertical"
    },
    {
      "description": "A large, flat-roofed skyscraper with a simpler, more rectilinear design compared to the central-right building. Its facade also features windows, but they are less prominent due to the building's angle and distance.",
      "location": "left foreground",
      "relationship": "It stands to the left of the central skyscraper, partially obscuring other buildings behind it and contributing to the layered effect.",
      "relative_size": "large within frame",
      "shape_and_color": "Large, flat, angular forms, dark grey to black.",
      "texture": "Rough concrete texture on the upper sections, smoother on the windowed parts.",
      "appearance_details": "The building has a distinct overhang or cantilevered section at the top-left, adding architectural interest.",
      "orientation": "diagonal, leaning towards the right"
    },
    {
      "description": "A cluster of mid-rise buildings, less distinct than the foreground structures, with visible window patterns.",
      "location": "midground, behind the foreground buildings",
      "relationship": "These buildings provide depth and context, suggesting a dense urban environment behind the prominent foreground structures.",
      "relative_size": "medium",
      "shape_and_color": "Rectangular forms, dark grey with hints of white from windows.",
      "texture": "Less defined, appearing smoother due to distance.",
      "appearance_details": "Their details are softened by distance and atmospheric haze.",
      "number_of_objects": 3,
      "orientation": "vertical"
    },
    {
      "description": "A tall, slender skyscraper with a facade composed of many small, closely spaced rectangular windows, creating a dense, textured pattern.",
      "location": "bottom-right midground",
      "relationship": "It stands slightly behind and to the right of the main central skyscraper, adding to the overall height of the urban landscape.",
      "relative_size": "medium",
      "shape_and_color": "Tall, slender, rectangular, dark grey with bright white window reflections.",
      "texture": "Densely packed, creating a fine-grained texture from the windows.",
      "appearance_details": "Its vertical lines are emphasized by the window arrangement.",
      "orientation": "vertical"
    }
  ],
  "background_setting": "A bright, uniformly white sky, indicative of an overcast day or a foggy atmosphere, which provides a stark contrast to the dark buildings.",
  "lighting": {
    "conditions": "overcast daylight",
    "direction": "diffused from above",
    "shadows": "soft, subtle shadows on the buildings, mainly emphasizing their geometric forms rather than sharp lines"
  },
  "aesthetics": {
    "composition": "dynamic and asymmetrical, with buildings angled and overlapping, creating a sense of depth and scale. The low angle emphasizes the height of the structures.",
    "color_scheme": "monochromatic (black and white) with a wide range of greys, emphasizing form and texture.",
    "mood_atmosphere": "grand, imposing, architectural, and slightly dramatic.",
    "aesthetic_score": "very high",
    "preference_score": "very high"
  },
  "photographic_characteristics": {
    "depth_of_field": "deep",
    "focus": "sharp focus on the foreground and midground buildings, with slight softening towards the background.",
    "camera_angle": "very low angle, looking upwards from ground level.",
    "lens_focal_length": "wide-angle"
  },
  "style_medium": "photograph",
  "text_render": [],
  "context": "An architectural photograph, likely intended for art prints, urban exploration photography, or a design magazine, focusing on the abstract beauty of modern cityscapes.",
  "artistic_style": "minimalist, high-contrast, geometric"
}
"""


@pytest.fixture
def vlm():
    return FiboVLM(model_id="briaai/FIBO-vlm")


def test_vlm_generate_json(vlm):
    # given
    input_prompt = "A hyper-detailed, ultra-fluffy owl sitting in the trees at night, looking directly at the camera with wide, adorable, expressive eyes. Its feathers are soft and voluminous, catching the cool moonlight with subtle silver highlights. The owl's gaze is curious and full of charm, giving it a whimsical, storybook-like personality."

    # when
    json_output = vlm.generate(prompt=input_prompt, seed=42)

    # then
    assert json_output == OWL_PROMPT, "Generated JSON output does not match expected output exactly."
    assert isinstance(json.loads(json_output), dict), "Output should be a valid JSON object"


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
