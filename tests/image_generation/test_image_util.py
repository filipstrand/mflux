import PIL.Image
import pytest

from mflux.post_processing.image_util import ImageUtil


@pytest.fixture
def test_image():
    # Create a simple test image
    return PIL.Image.new("RGB", (100, 80), color="blue")


def test_expand_image_with_pixels(test_image):
    # Test expanding with pixel values
    expanded = ImageUtil.expand_image(
        test_image,
        top=20,
        right=30,
        bottom=40,
        left=10,
        fill_color=(255, 0, 0),  # red
    )

    # Check dimensions
    assert expanded.width == 100 + 30 + 10  # original + right + left
    assert expanded.height == 80 + 20 + 40  # original + top + bottom

    # Check color at corners (should be the fill color)
    assert expanded.getpixel((0, 0)) == (255, 0, 0)  # top-left
    assert expanded.getpixel((139, 0)) == (255, 0, 0)  # top-right
    assert expanded.getpixel((0, 139)) == (255, 0, 0)  # bottom-left
    assert expanded.getpixel((139, 139)) == (255, 0, 0)  # bottom-right

    # Check original image is preserved in the middle
    assert expanded.getpixel((15, 25)) == (0, 0, 255)  # blue


def test_expand_image_with_percentages(test_image):
    # Test expanding with percentage values
    expanded = ImageUtil.expand_image(
        test_image,
        top="25%",
        right="30%",
        bottom="50%",
        left="10%",
        fill_color=(0, 255, 0),  # green
    )

    # Calculate expected dimensions
    expected_top = int(0.25 * 80)  # 25% of height
    expected_right = int(0.3 * 100)  # 30% of width
    expected_bottom = int(0.5 * 80)  # 50% of height
    expected_left = int(0.1 * 100)  # 10% of width

    expected_width = 100 + expected_right + expected_left
    expected_height = 80 + expected_top + expected_bottom

    # Check dimensions
    assert expanded.width == expected_width
    assert expanded.height == expected_height

    # Check fill color
    assert expanded.getpixel((0, 0)) == (0, 255, 0)


def test_expand_image_with_invalid_values(test_image):
    # Test with invalid percentage string (a string that does not end with %"
    with pytest.raises(ValueError):
        ImageUtil.expand_image(test_image, top="25#", right="30", bottom="50%", left="10")

    with pytest.raises(ValueError):
        ImageUtil.expand_image(test_image, top="25", right="30$", bottom="50%", left="10")

    with pytest.raises(ValueError):
        ImageUtil.expand_image(test_image, top="25", right="30", bottom="50#", left="10")

    with pytest.raises(ValueError):
        ImageUtil.expand_image(test_image, top="25", right="30", bottom="50#", left="10*")


def test_create_outpaint_mask_image():
    # Test with various padding values
    orig_width = 100
    orig_height = 80
    top_padding = 20
    right_padding = 30
    bottom_padding = 40
    left_padding = 10

    mask = ImageUtil.create_outpaint_mask_image(
        orig_width=orig_width,
        orig_height=orig_height,
        top=top_padding,
        right=right_padding,
        bottom=bottom_padding,
        left=left_padding,
    )

    # Check dimensions of the mask
    expected_width = orig_width + right_padding + left_padding
    expected_height = orig_height + bottom_padding + top_padding
    assert mask.width == expected_width
    assert mask.height == expected_height

    # Check padding areas are white (255, 255, 255)
    # Top-left corner
    assert mask.getpixel((5, 5)) == (255, 255, 255)

    # Top-right corner
    assert mask.getpixel((expected_width - 5, 5)) == (255, 255, 255)

    # Bottom-left corner
    assert mask.getpixel((5, expected_height - 5)) == (255, 255, 255)

    # Bottom-right corner
    assert mask.getpixel((expected_width - 5, expected_height - 5)) == (255, 255, 255)

    # Check center is black (0, 0, 0)
    center_x = left_padding + (orig_width // 2)
    center_y = top_padding + (orig_height // 2)
    assert mask.getpixel((center_x, center_y)) == (0, 0, 0)

    # Check boundaries
    # Top edge of center box
    assert mask.getpixel((left_padding + 10, top_padding)) == (0, 0, 0)

    # Right edge of center box
    assert mask.getpixel((left_padding + orig_width - 1, top_padding + 10)) == (0, 0, 0)

    # Bottom edge of center box
    assert mask.getpixel((left_padding + 10, top_padding + orig_height - 1)) == (0, 0, 0)

    # Left edge of center box
    assert mask.getpixel((left_padding, top_padding + 10)) == (0, 0, 0)
