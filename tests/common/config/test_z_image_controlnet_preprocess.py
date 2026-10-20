import cv2
import numpy as np
import PIL.Image
import pytest

from mflux.models.z_image.variants.controlnet.control_types import ControlType
from mflux.models.z_image.variants.controlnet.controlnet_util import ZImageControlnetUtil


def _photo() -> PIL.Image.Image:
    # A single bright rectangle on black gives the edge/line detectors something to find.
    a = np.zeros((64, 64, 3), dtype=np.uint8)
    a[16:48, 16:48] = 255
    return PIL.Image.fromarray(a)


@pytest.mark.fast
def _assert_is_hint(hint, img):
    arr = np.array(hint)
    assert hint.size == img.size
    assert hint.mode == "RGB"
    # a real hint differs from the input photo
    assert arr.tobytes() != np.array(img).tobytes()
    # and actually contains detected strokes: an all-black output (detector disabled) must fail
    assert arr.max() == 255
    assert (arr == 255).any()


def test_canny_produces_a_hint_not_the_original():
    img = _photo()
    _assert_is_hint(ZImageControlnetUtil._preprocess(img, ControlType.canny), img)


@pytest.mark.skipif(
    not hasattr(cv2, "createLineSegmentDetector"),
    reason="opencv-python 4.1-4.7 does not ship LineSegmentDetector",
)
def test_mlsd_produces_a_hint_not_the_original():
    img = _photo()
    mlsd = ZImageControlnetUtil._preprocess(img, ControlType.mlsd)
    _assert_is_hint(mlsd, img)
    # the mlsd hint is line strokes on black, so most of it is black
    assert (np.array(mlsd) == 0).mean() > 0.5


@pytest.mark.parametrize("shape", [(2, 1, 4), (2, 4)], ids=["opencv4", "opencv5"])
def test_mlsd_accepts_both_opencv_line_shapes(monkeypatch, shape):
    # OpenCV 4.x returns LSD lines shaped (N, 1, 4); OpenCV 5.x returns (N, 4).
    lines = np.array([[8, 8, 56, 8], [8, 56, 56, 56]], dtype=np.float32).reshape(shape)

    class _Detector:
        def detect(self, _image):
            return lines, None, None, None

    monkeypatch.setattr(cv2, "createLineSegmentDetector", lambda: _Detector(), raising=False)
    mlsd = np.array(ZImageControlnetUtil._preprocess(_photo(), ControlType.mlsd))
    assert (mlsd[8, 8:57] == 255).all()
    assert (mlsd[56, 8:57] == 255).all()
    assert (mlsd == 0).mean() > 0.5
