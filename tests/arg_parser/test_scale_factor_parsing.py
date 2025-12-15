from unittest.mock import patch

import pytest

from mflux.cli.parser.parsers import CommandLineParser, int_or_special_value
from mflux.utils.scale_factor import ScaleFactor


def _create_generic_scale_parser() -> CommandLineParser:
    parser = CommandLineParser(description="Generic parser for testing ScaleFactor")
    parser.add_argument("--dim", type=int_or_special_value, help="A dimension that supports ScaleFactor")
    return parser


@pytest.fixture
def generic_parser() -> CommandLineParser:
    return _create_generic_scale_parser()


@pytest.mark.fast
def test_scale_factor_auto(generic_parser):
    with patch("sys.argv", ["test", "--dim", "auto"]):
        args = generic_parser.parse_args()
        assert isinstance(args.dim, ScaleFactor)
        assert args.dim.value == 1


@pytest.mark.fast
def test_scale_factor_multiplier_format(generic_parser):
    # Test integer scale factor
    with patch("sys.argv", ["test", "--dim", "2x"]):
        args = generic_parser.parse_args()
        assert isinstance(args.dim, ScaleFactor)
        assert args.dim.value == 2

    # Test float scale factor
    with patch("sys.argv", ["test", "--dim", "1.5x"]):
        args = generic_parser.parse_args()
        assert isinstance(args.dim, ScaleFactor)
        assert args.dim.value == 1.5


@pytest.mark.fast
def test_plain_integer_dimensions(generic_parser):
    with patch("sys.argv", ["test", "--dim", "1024"]):
        args = generic_parser.parse_args()
        assert isinstance(args.dim, int)
        assert args.dim == 1024


@pytest.mark.fast
def test_invalid_scale_factor_format(generic_parser):
    # Invalid format without 'x'
    with patch("sys.argv", ["test", "--dim", "2.5"]):
        with pytest.raises(SystemExit):
            generic_parser.parse_args()

    # Invalid format with multiple 'x'
    with patch("sys.argv", ["test", "--dim", "2xx"]):
        with pytest.raises(SystemExit):
            generic_parser.parse_args()

    # Invalid non-numeric value
    with patch("sys.argv", ["test", "--dim", "abcx"]):
        with pytest.raises(SystemExit):
            generic_parser.parse_args()


@pytest.mark.fast
def test_case_insensitive_scale_factor(generic_parser):
    with patch("sys.argv", ["test", "--dim", "2X"]):
        args = generic_parser.parse_args()
        assert isinstance(args.dim, ScaleFactor)
        assert args.dim.value == 2
