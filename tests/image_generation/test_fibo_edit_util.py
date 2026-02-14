import json

import pytest
from PIL import Image

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
