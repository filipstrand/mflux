from unittest.mock import MagicMock, patch

import pytest

from mflux.callbacks.instances.battery_saver import BatterySaver, get_battery_percentage
from mflux.error.exceptions import StopImageGenerationException


def test_get_battery_percentage_while_charging():
    """Test that the function returns None when the output doesn't match the expected pattern."""
    with patch("subprocess.run") as mock_run:
        # Set up mock to return an output that doesn't match the expected pattern
        mock_result = MagicMock()
        mock_result.stdout = "Now drawing from 'AC Power'"
        mock_run.return_value = mock_result

        # Call the function
        percentage = get_battery_percentage()

        # Assert the function returns None when no match is found
        assert percentage is None


def test_battery_saver_below_limit():
    """Test that BatterySaver raises an exception when the battery is below the limit."""
    with patch("mflux.callbacks.instances.battery_saver.get_battery_percentage") as mock_get:
        # Configure mock to return a battery percentage below the limit
        mock_get.return_value = 5

        # Create a BatterySaver instance with a limit of 10%
        battery_saver = BatterySaver(battery_percentage_stop_limit=10)

        # Assert that calling before_loop raises StopImageGenerationException
        with pytest.raises(StopImageGenerationException) as excinfo:
            battery_saver.call_before_loop()

        # Check the exception message contains the correct limit and percentage
        assert "10%" in str(excinfo.value)
        assert "5%" in str(excinfo.value)


def test_battery_saver_above_limit():
    """Test that BatterySaver does not raise an exception when the battery is above the limit."""
    with patch("mflux.callbacks.instances.battery_saver.get_battery_percentage") as mock_get:
        # Configure mock to return a battery percentage above the limit
        mock_get.return_value = 20

        # Create a BatterySaver instance with a limit of 10%
        battery_saver = BatterySaver(battery_percentage_stop_limit=10)

        # Assert that calling before_loop does not raise an exception
        battery_saver.call_before_loop()


def test_battery_saver_none_percentage():
    """Test that BatterySaver does not raise an exception when percentage is None."""
    with patch("mflux.callbacks.instances.battery_saver.get_battery_percentage") as mock_get:
        # Configure mock to return None (e.g., on non-battery systems)
        mock_get.return_value = None

        # Create a BatterySaver instance
        battery_saver = BatterySaver()

        # Assert that calling before_loop does not raise an exception when percentage is None
        battery_saver.call_before_loop()


def test_battery_saver_equal_to_limit():
    """Test that BatterySaver raises an exception when the battery equals the limit."""
    with patch("mflux.callbacks.instances.battery_saver.get_battery_percentage") as mock_get:
        # Configure mock to return a battery percentage equal to the limit
        mock_get.return_value = 10

        # Create a BatterySaver instance with a limit of 10%
        battery_saver = BatterySaver(battery_percentage_stop_limit=10)

        # Assert that calling before_loop raises StopImageGenerationException
        with pytest.raises(StopImageGenerationException) as excinfo:
            battery_saver.call_before_loop()

        # Check the exception message contains the correct limit and percentage
        assert "10%" in str(excinfo.value)
