import pytest

from mflux.ui.scale_factor import ScaleFactor, parse_scale_factor


def test_scale_factor_init():
    """Test ScaleFactor initialization and validation"""
    # Valid integer scale factor
    sf = ScaleFactor(value=2)
    assert sf.value == 2

    # Valid float scale factor
    sf = ScaleFactor(value=1.5)
    assert sf.value == 1.5

    # Zero should raise ValueError
    with pytest.raises(ValueError, match="Scale factor must be positive"):
        ScaleFactor(value=0)

    # Negative should raise ValueError
    with pytest.raises(ValueError, match="Scale factor must be positive"):
        ScaleFactor(value=-1)


def test_scale_factor_get_scaled_value():
    """Test ScaleFactor.get_scaled_value method"""
    # Test with non-perfect multiple (needs rounding down)
    sf = ScaleFactor(value=1.5)
    assert sf.get_scaled_value(100) == 144  # 1.5 * 100 - (1.5 * 100) % 16 = 150 - 6 = 144

    # Test with scale factor that creates remainder
    sf = ScaleFactor(value=1.2)
    assert sf.get_scaled_value(100) == 112  # 1.2 * 100 - (1.2 * 100) % 16 = 120 - 8 = 112

    # Test with larger remainder
    sf = ScaleFactor(value=1.1)
    assert sf.get_scaled_value(200) == 208  # 1.1 * 200 - (1.1 * 200) % 16 = 220 - 12 = 208

    # Test with custom pixel_steps
    sf = ScaleFactor(value=1.3)
    assert sf.get_scaled_value(100, pixel_steps=32) == 128  # 1.3 * 100 - (1.3 * 100) % 32 = 130 - 2 = 128

    # Test edge case where result would be less than pixel_steps
    sf = ScaleFactor(value=0.1)
    assert sf.get_scaled_value(100) == 0  # 0.1 * 100 - (0.1 * 100) % 16 = 10 - 10 = 0


def test_parse_scale_factor_valid():
    """Test parsing valid scale factor strings"""
    # Integer scale factors
    sf = parse_scale_factor("1x")
    assert sf.value == 1

    sf = parse_scale_factor("2x")
    assert sf.value == 2

    sf = parse_scale_factor("10x")
    assert sf.value == 10

    # Float scale factors
    sf = parse_scale_factor("1.5x")
    assert sf.value == 1.5

    sf = parse_scale_factor("2.75X")
    assert sf.value == 2.75

    # With whitespace
    sf = parse_scale_factor("  2x  ")
    assert sf.value == 2


def test_parse_scale_factor_invalid():
    """Test parsing invalid scale factor strings"""
    # Missing 'x'
    with pytest.raises(ValueError, match="Invalid scale factor format"):
        parse_scale_factor("2")

    # Multiple 'x'
    with pytest.raises(ValueError, match="Invalid scale factor format"):
        parse_scale_factor("2xx")

    # Non-numeric value
    with pytest.raises(ValueError, match="Invalid scale factor format"):
        parse_scale_factor("abcx")

    # Empty before 'x'
    with pytest.raises(ValueError, match="Invalid scale factor format"):
        parse_scale_factor("x")

    # Invalid format
    with pytest.raises(ValueError, match="Invalid scale factor format"):
        parse_scale_factor("2.5.5x")

    # Negative values should fail at parsing
    with pytest.raises(ValueError, match="Invalid scale factor format"):
        parse_scale_factor("-1x")

    # Zero should parse but fail in ScaleFactor init
    with pytest.raises(ValueError, match="Scale factor must be positive"):
        parse_scale_factor("0x")


def test_scale_factor_realistic_dimensions():
    """Test scale factor with realistic image dimensions"""
    # 2x upscale of 512x512 image
    sf = ScaleFactor(value=2)
    assert sf.get_scaled_value(512) == 1024  # 2 * 512 - (2 * 512) % 16 = 1024 - 0 = 1024

    # 1.5x upscale of 768x768 image
    sf = ScaleFactor(value=1.5)
    assert sf.get_scaled_value(768) == 1152  # 1.5 * 768 - (1.5 * 768) % 16 = 1152 - 0 = 1152

    # 3x upscale of 256x256 image
    sf = ScaleFactor(value=3)
    assert sf.get_scaled_value(256) == 768  # 3 * 256 - (3 * 256) % 16 = 768 - 0 = 768

    # 0.5x downscale of 1024x1024 image
    sf = ScaleFactor(value=0.5)
    assert sf.get_scaled_value(1024) == 512  # 0.5 * 1024 - (0.5 * 1024) % 16 = 512 - 0 = 512
