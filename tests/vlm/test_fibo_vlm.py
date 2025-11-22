import json
from pathlib import Path

import pytest
from PIL import Image

from mflux.models.fibo_vlm.model.fibo_vlm import FiboVLM


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
    expected_json_output = '{"short_description":"A hyper-detailed, ultra-fluffy owl sitting in the trees at night, looking directly at the camera with wide, adorable, expressive eyes. Its feathers are soft and voluminous, catching the cool moonlight with subtle silver highlights. The owl\'s gaze is curious and full of charm, giving it a whimsical, storybook-like personality.","objects":[{"description":"An adorable, fluffy owl with large, expressive eyes and soft, voluminous feathers. Its plumage is a mix of warm browns, grays, and subtle silver highlights from the moonlight.","location":"center","relationship":"The owl is the sole subject, perched comfortably within its environment.","relative_size":"large within frame","shape_and_color":"Round head, large eyes, bulky body, predominantly brown and grey with silver accents.","texture":"Extremely soft, fluffy, and detailed feathers, giving a plush toy-like appearance.","appearance_details":"The eyes are wide, dark, and reflective, conveying a sense of wonder and curiosity. The beak is small and light-colored, almost hidden by the feathers. Subtle silver highlights catch the moonlight on its feathers.","orientation":"upright, facing forward"}],"background_setting":"A dark, nocturnal forest setting with blurred trees and foliage, illuminated by a soft, cool moonlight. The background is out of focus, emphasizing the owl.","lighting":{"conditions":"moonlight","direction":"backlit and side-lit from the left","shadows":"soft, diffused shadows on the right side of the owl and within the background foliage, indicating a single light source."},"aesthetics":{"composition":"centered, portrait composition","color_scheme":"cool blues and silvers from the moonlight contrasting with warm browns and grays of the owl and forest.","mood_atmosphere":"mysterious, enchanting, whimsical, and serene.","aesthetic_score":"very high","preference_score":"very high"},"photographic_characteristics":{"depth_of_field":"shallow","focus":"sharp focus on the owl\'s face and eyes, with a soft blur in the background.","camera_angle":"eye-level","lens_focal_length":"portrait lens (e.g., 50mm-85mm)"},"style_medium":"digital illustration","text_render":[],"context":"A whimsical character illustration, possibly for a children\'s book, animated film, or fantasy art collection.","artistic_style":"fantasy, illustrative, detailed"}'
    assert json_output == expected_json_output, "Generated JSON output does not match expected output exactly."
    json_obj = json.loads(json_output)
    assert isinstance(json_obj, dict), "Output should be a valid JSON object"


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
    expected_json_output = '{"short_description":"A charming, stylized illustration of a young owl perched on a mossy branch in a dimly lit forest. The owl has large, expressive eyes and soft, feathered textures. The background is a blur of dark trees and foliage, creating a serene and slightly mysterious atmosphere. The overall impression is one of innocence and nature\'s quiet beauty.","objects":[{"description":"A young owl with large, round, dark eyes that have bright yellow irises. Its plumage is a soft, mottled brown and beige, with distinct feather patterns. It has small, pointed ear tufts and a small yellow beak.","location":"center","relationship":"perched on a branch","relative_size":"medium","shape_and_color":"Rounded body shape, predominantly brown and beige with yellow eyes and beak.","texture":"soft, feathery","appearance_details":"Its eyes are wide and seem to gaze directly at the viewer. The feathers have a layered, textured appearance.","number_of_objects":1,"pose":"Sitting upright, with wings slightly tucked.","expression":"curious and gentle","action":"perching","gender":"neutral","orientation":"upright"},{"description":"A thick, gnarled tree branch covered in vibrant green moss.","location":"bottom-center foreground","relationship":"the owl is perched on this branch","relative_size":"medium","shape_and_color":"Irregular, organic shape, dark brown wood with bright green moss.","texture":"rough bark under soft, velvety moss","appearance_details":"The moss is lush and covers a significant portion of the branch\'s surface.","number_of_objects":1,"orientation":"horizontal"},{"description":"Several slender tree trunks and branches forming the background.","location":"background","relationship":"surrounding the owl and branch","relative_size":"large","shape_and_color":"Vertical, irregular shapes, dark brown and grey.","texture":"rough bark","appearance_details":"Some branches have sparse green leaves. The trees are out of focus.","orientation":"vertical"}],"background_setting":"A dense, dark forest with tall trees and foliage. The background is softly blurred, suggesting depth and a sense of enclosure. Hints of lighter tones suggest distant light filtering through the canopy.","lighting":{"conditions":"dim forest light","direction":"soft, diffused light from the front and slightly above","shadows":"soft, subtle shadows that enhance the form of the owl and branch without being harsh"},"aesthetics":{"composition":"centered composition, with the owl as the clear focal point","color_scheme":"earthy tones of brown, beige, and green, with contrasting yellow accents and a dark, muted background","mood_atmosphere":"serene, innocent, slightly mysterious","aesthetic_score":"high","preference_score":"high"},"photographic_characteristics":{"depth_of_field":"shallow, with a blurred background","focus":"sharp focus on the owl and the branch","camera_angle":"eye-level","lens_focal_length":"standard lens (e.g., 50mm)"},"style_medium":"digital illustration","text_render":[],"context":"This image would be suitable for children\'s book illustrations, nature-themed art, or as a decorative element for a calm, natural setting.","artistic_style":"stylized, soft, detailed"}'
    assert json_output == expected_json_output, "Generated JSON output does not match expected output exactly."
    json_obj = json.loads(json_output)
    assert isinstance(json_obj, dict), "Output should be a valid JSON object"


def test_vlm_refine_json_text_only(vlm):
    # given
    structured_prompt = '{"short_description":"A hyper-detailed, ultra-fluffy owl sitting in the trees at night, looking directly at the camera with wide, adorable, expressive eyes. Its feathers are soft and voluminous, catching the cool moonlight with subtle silver highlights. The owl\'s gaze is curious and full of charm, giving it a whimsical, storybook-like personality.","objects":[{"description":"An adorable, fluffy owl with large, expressive eyes and soft, voluminous feathers. Its plumage is a mix of warm browns, grays, and subtle silver highlights from the moonlight.","location":"center","relationship":"The owl is the sole subject, perched comfortably within its environment.","relative_size":"large within frame","shape_and_color":"Round head, large eyes, bulky body, predominantly brown and grey with silver accents.","texture":"Extremely soft, fluffy, and detailed feathers, giving a plush toy-like appearance.","appearance_details":"The eyes are wide, dark, and reflective, conveying a sense of wonder and curiosity. The beak is small and light-colored, almost hidden by the feathers. Subtle silver highlights catch the moonlight on its feathers.","orientation":"upright, facing forward"}],"background_setting":"A dark, nocturnal forest setting with blurred trees and foliage, illuminated by a soft, cool moonlight. The background is out of focus, emphasizing the owl.","lighting":{"conditions":"moonlight","direction":"backlit and side-lit from the left","shadows":"soft, diffused shadows on the right side of the owl and within the background foliage, indicating a single light source."},"aesthetics":{"composition":"centered, portrait composition","color_scheme":"cool blues and silvers from the moonlight contrasting with warm browns and grays of the owl and forest.","mood_atmosphere":"mysterious, enchanting, whimsical, and serene.","aesthetic_score":"very high","preference_score":"very high"},"photographic_characteristics":{"depth_of_field":"shallow","focus":"sharp focus on the owl\'s face and eyes, with a soft blur in the background.","camera_angle":"eye-level","lens_focal_length":"portrait lens (e.g., 50mm-85mm)"},"style_medium":"digital illustration","text_render":[],"context":"A whimsical character illustration, possibly for a children\'s book, animated film, or fantasy art collection.","artistic_style":"fantasy, illustrative, detailed"}'
    editing_instructions = "make the owl white color but keep everything else exactly the same"

    # when
    json_output = vlm.generate(
        top_p=0.9,
        temperature=0.2,
        max_tokens=4096,
        stop=["<|im_end|>", "<|end_of_text|>"],
        task="refine",
        structured_prompt=structured_prompt,
        editing_instructions=editing_instructions,
        seed=42,
    )

    # then
    expected_json_output = '{"short_description":"A hyper-detailed, ultra-fluffy owl sitting in the trees at night, looking directly at the camera with wide, adorable, expressive eyes. Its feathers are soft and voluminous, catching the cool moonlight with subtle silver highlights. The owl\'s gaze is curious and full of charm, giving it a whimsical, storybook-like personality.","objects":[{"description":"An adorable, fluffy owl with large, expressive eyes and soft, voluminous feathers. Its plumage is a mix of white and subtle silver highlights from the moonlight.","location":"center","relationship":"The owl is the sole subject, perched comfortably within its environment.","relative_size":"large within frame","shape_and_color":"Round head, large eyes, bulky body, predominantly white with silver accents.","texture":"Extremely soft, fluffy, and detailed feathers, giving a plush toy-like appearance.","appearance_details":"The eyes are wide, dark, and reflective, conveying a sense of wonder and curiosity. The beak is small and light-colored, almost hidden by the feathers. Subtle silver highlights catch the moonlight on its feathers.","orientation":"upright, facing forward"}],"background_setting":"A dark, nocturnal forest setting with blurred trees and foliage, illuminated by a soft, cool moonlight. The background is out of focus, emphasizing the owl.","lighting":{"conditions":"moonlight","direction":"backlit and side-lit from the left","shadows":"soft, diffused shadows on the right side of the owl and within the background foliage, indicating a single light source."},"aesthetics":{"composition":"centered, portrait composition","color_scheme":"cool blues and silvers from the moonlight contrasting with white of the owl and forest.","mood_atmosphere":"mysterious, enchanting, whimsical, and serene.","aesthetic_score":"very high","preference_score":"very high"},"photographic_characteristics":{"depth_of_field":"shallow","focus":"sharp focus on the owl\'s face and eyes, with a soft blur in the background.","camera_angle":"eye-level","lens_focal_length":"portrait lens (e.g., 50mm-85mm)"},"style_medium":"digital illustration","text_render":[],"context":"A whimsical character illustration, possibly for a children\'s book, animated film, or fantasy art collection.","artistic_style":"fantasy, illustrative, detailed"}'
    assert json_output == expected_json_output, "Generated JSON output does not match expected output exactly."
    json_obj = json.loads(json_output)
    assert isinstance(json_obj, dict), "Output should be a valid JSON object"
