import mlx.core as mx
import numpy as np
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


def test_binarize():
    # Create test data using numpy arrays and convert to mx arrays
    # Create a gradient array from 0 to 1
    gradient = np.linspace(0, 1, 10).reshape(1, 1, 1, 10).astype(np.float32)
    gradient_mx = mx.array(gradient)

    # Apply binarization
    result = ImageUtil._binarize(gradient_mx)

    # Expected: values < 0.5 should be 0, values >= 0.5 should be 1
    expected = mx.where(gradient_mx < 0.5, mx.zeros_like(gradient_mx), mx.ones_like(gradient_mx))

    # Convert results to numpy for easier comparison
    result_np = np.array(result)
    expected_np = np.array(expected)

    # Check values
    assert np.array_equal(result_np, expected_np)

    # Specifically check that the first 5 values are 0 and the rest are 1
    assert np.all(result_np[:, :, :, :5] == 0)
    assert np.all(result_np[:, :, :, 5:] == 1)


def test_to_array_with_mask():
    # Create a test image with gradient colors
    from PIL import Image, ImageDraw

    # Create a simple mask image (white square on black background)
    mask_img = Image.new("RGB", (100, 100), color="black")
    draw = ImageDraw.Draw(mask_img)
    draw.rectangle((25, 25, 75, 75), fill="white")

    # Convert to array with is_mask=False (should normalize)
    regular_array = ImageUtil.to_array(mask_img, is_mask=False)

    # Convert to array with is_mask=True (should binarize)
    mask_array = ImageUtil.to_array(mask_img, is_mask=True)

    # Check shapes
    assert regular_array.shape == mask_array.shape

    # Regular array should have normalized values between -1 and 1
    assert mx.min(regular_array) < 0
    assert mx.max(regular_array) <= 1.0

    # Mask array should only have binary values (0 or 1)
    unique_values = set(mx.array.flatten(mask_array).tolist())
    assert unique_values == {0.0, 1.0} or unique_values == {0.0} or unique_values == {1.0}

    # The center of the mask should be 1 (white square)
    assert mask_array[0, 0, 50, 50] == 1.0

    # The corners should be 0 (black background)
    assert mask_array[0, 0, 0, 0] == 0.0
    assert mask_array[0, 0, 0, 99] == 0.0
    assert mask_array[0, 0, 99, 0] == 0.0
    assert mask_array[0, 0, 99, 99] == 0.0
