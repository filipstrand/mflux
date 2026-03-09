import json
from pathlib import Path

import pytest
from PIL import Image

from mflux.models.fibo_vlm.model.fibo_vlm import FiboVLM
from tests.image_generation.test_generate_image_fibo import OWL_PROMPT, OWL_PROMPT_REFINED

VLM_GENERATE_PROMPT = """
{
  "short_description": "A hyper-detailed, ultra-fluffy owl sitting in the trees at night, looking directly at the camera with wide, adorable, expressive eyes. Its feathers are soft and voluminous, catching the cool moonlight with subtle silver highlights. The owl's gaze is curious and full of charm, giving it a whimsical, storybook-like personality.",
  "objects": [
    {
      "description": "An adorable, fluffy owl with large, expressive eyes and soft, voluminous feathers. Its plumage is a mix of earthy tones with subtle silver highlights from the moonlight.",
      "location": "center",
      "relationship": "The owl is the sole subject, perched comfortably within its environment.",
      "relative_size": "large within frame",
      "shape_and_color": "Round head, large eyes, voluminous body shape. Predominantly earthy browns, grays, and subtle silver.",
      "texture": "soft, fluffy, voluminous feathers",
      "appearance_details": "Wide, dark pupils in large, light-colored eyes. Small, delicate beak. Feathers have a layered, textured appearance.",
      "number_of_objects": 1,
      "pose": "Sitting upright, facing forward.",
      "expression": "curious and gentle",
      "action": "looking directly at the viewer",
      "gender": "unknown",
      "orientation": "upright"
    }
  ],
  "background_setting": "A dense, dark forest at night. The trees are silhouetted against a dark sky, with subtle hints of moonlight filtering through the leaves. The overall environment is mysterious and serene.",
  "lighting": {
    "conditions": "night, moonlight",
    "direction": "side-lit from the left, with some ambient light from above",
    "shadows": "soft, elongated shadows cast by the owl and branches, adding depth"
  },
  "aesthetics": {
    "composition": "centered framing, with the owl as the clear focal point",
    "color_scheme": "cool, muted tones of blues, grays, and deep greens, with subtle silver highlights",
    "mood_atmosphere": "mysterious, serene, whimsical, enchanting",
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
  "context": "This image is a whimsical illustration, suitable for a children's book, a fantasy game, or a decorative piece evoking a sense of magic and wonder.",
  "artistic_style": "storybook, fantasy, detailed"
}
"""

INSPIRE_PROMPT = """
{
  "short_description": "A charming, stylized illustration of a young owl perched on a thick, gnarled tree branch. The owl is facing forward with large, expressive eyes and its wings are slightly tucked. The background depicts a dark, mystical forest with silhouetted trees and a subtle, glowing moon or light source, creating an enchanting and slightly mysterious atmosphere. The overall aesthetic is whimsical and serene.",
  "objects": [
    {
      "description": "A young, fluffy owl with large, round, dark eyes and prominent ear tufts. Its plumage is a mix of soft grays, whites, and browns, with detailed feather patterns.",
      "location": "center",
      "relationship": "perched on a tree branch",
      "relative_size": "medium within frame",
      "shape_and_color": "Round body, large eyes, predominantly grey and brown.",
      "texture": "soft, feathery",
      "appearance_details": "Its gaze is direct and engaging. The beak is small and yellow.",
      "number_of_objects": 1,
      "pose": "Sitting upright, facing forward.",
      "expression": "curious and calm",
      "action": "perching",
      "gender": "neutral",
      "orientation": "upright"
    },
    {
      "description": "A thick, ancient tree branch with a rough, textured surface. It is dark brown and appears sturdy, providing a perch for the owl.",
      "location": "bottom-center foreground",
      "relationship": "supports the owl",
      "relative_size": "large",
      "shape_and_color": "Irregular, thick, dark brown.",
      "texture": "rough, bark-like",
      "appearance_details": "The branch has visible knots and grooves.",
      "number_of_objects": 1,
      "orientation": "horizontal"
    },
    {
      "description": "A cluster of stylized mushrooms with rounded caps and thin stems, appearing in muted yellow and brown tones.",
      "location": "bottom-right midground",
      "relationship": "growing near the base of a tree trunk",
      "relative_size": "small",
      "shape_and_color": "Rounded caps, thin stems, yellow and brown.",
      "texture": "smooth caps, slightly fibrous stems",
      "appearance_details": "They are small and somewhat sparse.",
      "number_of_objects": 3,
      "orientation": "vertical"
    },
    {
      "description": "A large, dark tree trunk on the right side of the frame, with a rough bark texture.",
      "location": "right midground",
      "relationship": "part of the forest environment, framing the scene",
      "relative_size": "large",
      "shape_and_color": "Vertical, dark brown.",
      "texture": "rough bark",
      "appearance_details": "It is partially obscured by shadow.",
      "number_of_objects": 1,
      "orientation": "vertical"
    }
  ],
  "background_setting": "A dense, dark forest with tall, silhouetted trees. Bare branches create a complex pattern against a deep blue, moonlit sky. The ground is covered in dark foliage and scattered leaves.",
  "lighting": {
    "conditions": "moonlit night",
    "direction": "backlit and side-lit from the upper left",
    "shadows": "soft, elongated shadows cast by the trees and branches"
  },
  "aesthetics": {
    "composition": "centered framing of the owl, with the surrounding forest creating a natural frame",
    "color_scheme": "cool color palette dominated by blues, grays, and dark browns, with subtle highlights",
    "mood_atmosphere": "mysterious, enchanting, serene",
    "aesthetic_score": "high",
    "preference_score": "high"
  },
  "photographic_characteristics": {
    "depth_of_field": "shallow, with the background softly blurred",
    "focus": "sharp focus on the owl",
    "camera_angle": "eye-level",
    "lens_focal_length": "standard lens"
  },
  "style_medium": "digital illustration",
  "text_render": [],
  "context": "This image is suitable for a children's book illustration, a fantasy-themed graphic, or a decorative piece for a nature enthusiast.",
  "artistic_style": "stylized, whimsical, detailed"
}
"""

SKYSCRAPERS_INSPIRE_PROMPT = """
{
  "short_description": "A dramatic, low-angle shot of several modern skyscrapers in black and white, emphasizing their towering height and geometric forms against a bright, overcast sky. The buildings feature repetitive window patterns and varying architectural details, creating a sense of urban grandeur and perspective.",
  "objects": [
    {
      "description": "A tall skyscraper with a facade of numerous rectangular windows, arranged in a grid pattern. The building has a slightly curved or angled design, giving it a dynamic appearance.",
      "location": "center-right foreground",
      "relationship": "This is the most prominent building, drawing the viewer's eye upwards.",
      "relative_size": "large within frame",
      "shape_and_color": "Rectangular and angular forms, dark grey to black due to the black and white photography.",
      "texture": "Smooth concrete and glass, with visible window frames creating a textured grid.",
      "appearance_details": "The windows reflect the bright sky, appearing as light rectangles. Some sections of the building have protruding architectural elements.",
      "orientation": "vertical"
    },
    {
      "description": "A large, flat-roofed skyscraper with a distinct geometric pattern on its facade, possibly representing a different architectural style or material.",
      "location": "top-left to mid-left",
      "relationship": "It stands to the left and slightly behind the central skyscraper, adding depth to the composition.",
      "relative_size": "large within frame",
      "shape_and_color": "Angular and blocky forms, dark grey to black.",
      "texture": "Rough, possibly concrete or stone, with a textured geometric pattern.",
      "appearance_details": "The pattern consists of small, recessed rectangular shapes, creating a tactile surface.",
      "orientation": "vertical"
    },
    {
      "description": "A series of less distinct skyscrapers and taller structures receding into the background, their forms becoming more uniform and abstract.",
      "location": "mid-ground to background, spanning across the frame",
      "relationship": "These buildings provide context and scale, indicating a dense urban environment.",
      "relative_size": "medium to small",
      "shape_and_color": "Vertical rectangular shapes, varying shades of grey.",
      "texture": "Smooth and reflective surfaces, less defined due to distance and focus.",
      "appearance_details": "Their windows appear as faint lines or grids, contributing to the sense of depth.",
      "number_of_objects": 5,
      "orientation": "vertical"
    }
  ],
  "background_setting": "A bright, uniformly white sky, indicative of an overcast day or a blown-out sky, which provides a stark contrast to the dark buildings.",
  "lighting": {
    "conditions": "overcast daylight",
    "direction": "diffused from above",
    "shadows": "soft, subtle shadows on the buildings, emphasizing their forms without harsh lines"
  },
  "aesthetics": {
    "composition": "dynamic low-angle composition, leading lines created by the building edges and window patterns, emphasizing height and perspective.",
    "color_scheme": "monochromatic (black and white) with a wide range of greys, from deep blacks in the buildings to bright whites in the sky.",
    "mood_atmosphere": "grand, imposing, architectural, urban, serene.",
    "aesthetic_score": "very high",
    "preference_score": "very high"
  },
  "photographic_characteristics": {
    "depth_of_field": "deep",
    "focus": "sharp focus on the foreground and mid-ground buildings, with slight softening in the far background.",
    "camera_angle": "very low angle (worm's eye view), looking upwards at the skyscrapers.",
    "lens_focal_length": "wide-angle lens, to capture the expansive height and scale of the buildings."
  },
  "style_medium": "photograph",
  "text_render": [],
  "context": "An architectural photograph, likely intended for art prints, urban exploration, or a photography portfolio, focusing on the abstract beauty and scale of modern cityscapes.",
  "artistic_style": "minimalist, high-contrast, architectural"
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
    assert json_output == VLM_GENERATE_PROMPT, "Generated JSON output does not match expected output exactly."
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
