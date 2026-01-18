import json
from pathlib import Path

import pytest
from PIL import Image

from mflux.models.fibo_vlm.model.fibo_vlm import FiboVLM
from tests.image_generation.test_generate_image_fibo import OWL_PROMPT, OWL_PROMPT_REFINED

INSPIRE_PROMPT = """
{
  "short_description": "A charming, stylized illustration of a young owl perched on a thick, gnarled tree branch. The owl is depicted with large, expressive eyes and soft, feathered textures. It is surrounded by a dense, atmospheric forest with tall, dark trees and a subtle hint of moonlight filtering through the canopy. The overall impression is one of innocence and nocturnal wonder, rendered in a painterly, fantasy style.",
  "objects": [
    {
      "description": "A young owl with large, round, dark eyes that have bright white highlights, giving it an innocent and curious expression. Its plumage is a soft, mottled grey and brown, with distinct feather patterns. It has small, sharp yellow talons and a pointed, yellow beak.",
      "location": "center",
      "relationship": "perched on a tree branch",
      "relative_size": "medium within frame",
      "shape_and_color": "round body, large head, predominantly grey and brown with yellow accents",
      "texture": "soft, feathery, detailed",
      "appearance_details": "Its wings are folded close to its body, and its feet are gripping the branch firmly.",
      "number_of_objects": 1,
      "pose": "Sitting upright, facing forward.",
      "expression": "innocent, curious",
      "action": "perching",
      "gender": "neutral",
      "skin_tone_and_texture": "N/A (owl)",
      "orientation": "upright"
    },
    {
      "description": "A thick, ancient tree branch, twisted and textured, providing a perch for the owl. It is dark brown with visible wood grain and some mossy or lichen-like growth in places.",
      "location": "bottom-center foreground",
      "relationship": "supports the owl",
      "relative_size": "large",
      "shape_and_color": "curved, irregular shape, dark brown",
      "texture": "rough, woody, textured",
      "appearance_details": "The branch extends across the lower part of the frame, with smaller twigs branching off.",
      "number_of_objects": 1,
      "orientation": "horizontal, slightly angled"
    },
    {
      "description": "A cluster of stylized, dark tree trunks forming a dense forest background. They are tall and slender, with a rough, dark texture.",
      "location": "background",
      "relationship": "surrounds the owl and branch",
      "relative_size": "large",
      "shape_and_color": "vertical, cylindrical shapes, dark brown and black",
      "texture": "rough, bark-like",
      "appearance_details": "Some trees have faint glowing elements, possibly representing distant lights or magical flora.",
      "orientation": "vertical"
    },
    {
      "description": "Silhouetted branches and foliage, rendered in dark tones, creating a sense of depth and enclosure within the forest.",
      "location": "midground and background",
      "relationship": "forms the forest environment",
      "relative_size": "medium to large",
      "shape_and_color": "varied organic shapes, dark blues, purples, and blacks",
      "texture": "soft, painterly",
      "appearance_details": "Some elements have a subtle glow, adding to the mystical atmosphere.",
      "orientation": "varied"
    }
  ],
  "background_setting": "A mystical, dark forest at night. Tall, ancient trees with gnarled branches dominate the scene, with a hint of a pale, glowing orb (possibly a moon or magical light source) visible through the canopy in the upper left.",
  "lighting": {
    "conditions": "dim, atmospheric night lighting",
    "direction": "soft, diffused light from the upper left, with some ambient glow from within the forest",
    "shadows": "soft, elongated shadows contributing to the depth and mood"
  },
  "aesthetics": {
    "composition": "centered composition with the owl as the primary focal point, balanced by the tree branch below and the forest backdrop",
    "color_scheme": "predominantly cool, dark tones with accents of yellow and soft glowing highlights",
    "mood_atmosphere": "enchanting, serene, mysterious, innocent",
    "aesthetic_score": "very high",
    "preference_score": "very high"
  },
  "photographic_characteristics": {
    "depth_of_field": "shallow, with the background softly blurred",
    "focus": "sharp focus on the owl and the branch",
    "camera_angle": "eye-level",
    "lens_focal_length": "standard lens (e.g., 50mm)"
  },
  "style_medium": "digital illustration",
  "text_render": [],
  "context": "This image is suitable for a children's book illustration, a fantasy-themed graphic, or a decorative piece evoking a sense of nature and magic.",
  "artistic_style": "painterly, fantasy, stylized"
}
"""

SKYSCRAPERS_INSPIRE_PROMPT = """
{
  "short_description": "A dramatic, low-angle shot of several modern skyscrapers in black and white, emphasizing their towering height and geometric forms against a bright, overcast sky. The buildings feature repetitive window patterns and varying architectural details, creating a sense of urban grandeur and scale.",
  "objects": [
    {
      "description": "A tall skyscraper with a facade of numerous rectangular windows, arranged in a grid pattern. The building has a slightly curved or angled top section.",
      "location": "center-right foreground",
      "relationship": "This is the most prominent building, dominating the right side of the frame and serving as a primary focal point.",
      "relative_size": "large within frame",
      "shape_and_color": "Rectangular and angular, dark grey to black with bright white window reflections.",
      "texture": "Smooth, reflective glass and concrete, with visible mullions.",
      "appearance_details": "The windows appear as bright rectangles, contrasting with the dark building material. Some sections show decorative architectural elements.",
      "orientation": "vertical"
    },
    {
      "description": "A tall, slender skyscraper with a facade composed of many small, rectangular windows, creating a textured, grid-like appearance.",
      "location": "center-left midground",
      "relationship": "It stands behind and to the left of the central-right skyscraper, contributing to the depth of the urban landscape.",
      "relative_size": "large within frame",
      "shape_and_color": "Tall, slender rectangle, dark grey with bright white window reflections.",
      "texture": "Fine, repetitive texture from the windows, appearing smooth from a distance.",
      "appearance_details": "The uniformity of its windows gives it a sleek, modern look.",
      "orientation": "vertical"
    },
    {
      "description": "A large, flat-roofed building with a distinctive overhang or cantilevered section at its top, featuring a row of smaller, recessed windows.",
      "location": "top-left foreground",
      "relationship": "It partially frames the upper left corner, creating a sense of enclosure and depth in relation to the other skyscrapers.",
      "relative_size": "large within frame",
      "shape_and_color": "Large, flat, angular shape, dark grey to black.",
      "texture": "Rough, concrete-like texture on the main body, with smoother glass on the recessed windows.",
      "appearance_details": "The overhang creates a strong shadow effect, adding to its architectural drama.",
      "orientation": "horizontal"
    },
    {
      "description": "A skyscraper with a facade featuring larger, rectangular windows that are not perfectly uniform, giving it a slightly more varied texture.",
      "location": "bottom-right foreground",
      "relationship": "It is partially visible at the bottom right, grounding the composition and adding to the sense of a dense urban environment.",
      "relative_size": "medium within frame",
      "shape_and_color": "Vertical rectangular shape, dark grey with bright white window reflections.",
      "texture": "Slightly varied, textured surface from the window frames and building material.",
      "appearance_details": "The windows reflect the bright sky, appearing almost blown out.",
      "orientation": "vertical"
    }
  ],
  "background_setting": "A bright, uniformly white sky, suggesting an overcast day or a foggy atmosphere, which provides a stark contrast to the dark buildings.",
  "lighting": {
    "conditions": "overcast daylight",
    "direction": "diffused from above",
    "shadows": "soft, subtle shadows within the architectural recesses, emphasizing form rather than harsh lines"
  },
  "aesthetics": {
    "composition": "dynamic low-angle composition, with buildings converging towards the top-center, creating leading lines and a sense of immense height.",
    "color_scheme": "monochromatic (black and white) with a wide range of greys, emphasizing form, texture, and shadow.",
    "mood_atmosphere": "grand, imposing, modern, and slightly dramatic.",
    "aesthetic_score": "very high",
    "preference_score": "very high"
  },
  "photographic_characteristics": {
    "depth_of_field": "deep",
    "focus": "sharp focus on all visible architectural details of the skyscrapers.",
    "camera_angle": "very low angle, looking upwards from street level.",
    "lens_focal_length": "wide-angle"
  },
  "style_medium": "photograph",
  "text_render": [],
  "context": "An architectural photograph, likely intended for art prints, urban exploration blogs, or promotional material for city tourism, focusing on the grandeur of modern cityscapes.",
  "artistic_style": "minimalist, high-contrast, geometric"
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
