import json
from types import SimpleNamespace

import pytest
from PIL import Image

from mflux.models.fibo.variants.edit import util as fibo_edit_util_module
from mflux.models.fibo.variants.edit.util import FiboEditUtil


def test_ensure_edit_instruction_uses_existing_value():
    prompt = json.dumps({"short_description": "owl", "edit_instruction": "make it white"})
    updated = FiboEditUtil.ensure_edit_instruction(prompt, edit_instruction="ignored")
    updated_dict = json.loads(updated)
    assert updated_dict["edit_instruction"] == "make it white"


def test_ensure_edit_instruction_injects_value_when_missing():
    prompt = json.dumps({"short_description": "owl"})
    updated = FiboEditUtil.ensure_edit_instruction(prompt, edit_instruction="add glasses")
    updated_dict = json.loads(updated)
    assert updated_dict["edit_instruction"] == "add glasses"


def test_ensure_edit_instruction_requires_value_when_missing():
    prompt = json.dumps({"short_description": "owl"})
    with pytest.raises(ValueError, match="edit_instruction"):
        FiboEditUtil.ensure_edit_instruction(prompt, edit_instruction=None)


def test_load_edit_image_raises_for_mask_size_mismatch(tmp_path):
    image_path = tmp_path / "image.png"
    mask_path = tmp_path / "mask.png"
    Image.new("RGB", (64, 64), (255, 255, 255)).save(image_path)
    Image.new("L", (32, 32), 255).save(mask_path)

    with pytest.raises(ValueError, match="Mask and image must have the same size"):
        FiboEditUtil.load_edit_image(image_path=image_path, width=64, height=64, mask_path=mask_path)


def test_get_json_prompt_for_edit_returns_existing_json():
    args = SimpleNamespace(prompt='{"short_description":"owl","edit_instruction":"make it white"}', prompt_file=None)

    prompt = FiboEditUtil.get_json_prompt_for_edit(args, quantize=None)

    assert json.loads(prompt)["edit_instruction"] == "make it white"


def test_get_json_prompt_for_edit_requires_prompt_input():
    args = SimpleNamespace(prompt=None, prompt_file=None, image_path="input.png", mask_path=None)

    with pytest.raises(ValueError, match="requires an edit instruction via --prompt/--prompt-file"):
        FiboEditUtil.get_json_prompt_for_edit(args, quantize=None)


def test_get_json_prompt_for_edit_requires_image_for_natural_language_prompt():
    args = SimpleNamespace(prompt="turn the owl white", prompt_file=None, image_path=None, mask_path=None)

    with pytest.raises(ValueError, match="requires --image-path"):
        FiboEditUtil.get_json_prompt_for_edit(args, quantize=None)


def test_get_json_prompt_for_edit_routes_image_and_mask_to_vlm(monkeypatch, tmp_path):
    image_path = tmp_path / "image.png"
    mask_path = tmp_path / "mask.png"
    Image.new("RGB", (64, 64), (10, 20, 30)).save(image_path)
    Image.new("L", (64, 64), 255).save(mask_path)

    captured = {}

    class _FakeVLM:
        def __init__(self, quantize):
            captured["quantize"] = quantize

        def edit(self, image, edit_instruction, use_mask, seed):
            captured["image"] = image
            captured["edit_instruction"] = edit_instruction
            captured["use_mask"] = use_mask
            captured["seed"] = seed
            return json.dumps({"short_description": "owl", "edit_instruction": "make it white"})

    monkeypatch.setattr(fibo_edit_util_module, "FiboVLM", _FakeVLM)
    args = SimpleNamespace(prompt="turn the owl white", prompt_file=None, image_path=image_path, mask_path=mask_path)

    prompt = FiboEditUtil.get_json_prompt_for_edit(args, quantize=8)

    assert json.loads(prompt)["edit_instruction"] == "make it white"
    assert captured["quantize"] == 8
    assert captured["edit_instruction"] == "turn the owl white"
    assert captured["use_mask"] is True
    assert captured["seed"] == 42
    assert captured["image"].getpixel((0, 0)) == (128, 128, 128)
