import PIL.Image
import pytest

from mflux.utils.image_util import ImageUtil

pytestmark = pytest.mark.fast


def test_save_image_raises_when_the_image_cannot_be_written(tmp_path, monkeypatch):
    def fail(self, *args, **kwargs):
        raise PermissionError("read-only directory")

    monkeypatch.setattr(PIL.Image.Image, "save", fail)

    with pytest.raises(PermissionError):
        ImageUtil.save_image(PIL.Image.new("RGB", (8, 8)), tmp_path / "out.png")


def test_save_image_keeps_the_image_when_only_the_metadata_fails(tmp_path, monkeypatch):
    def fail(metadata, path):
        raise ValueError("cannot embed")

    monkeypatch.setattr(ImageUtil, "_embed_metadata", staticmethod(fail))
    out = tmp_path / "out.png"

    ImageUtil.save_image(PIL.Image.new("RGB", (8, 8)), out, metadata={"seed": 7})

    assert out.exists()
