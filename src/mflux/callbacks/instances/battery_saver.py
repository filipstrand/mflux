import json
import logging
import re
import subprocess

from mflux.callbacks.callback import BeforeLoopCallback
from mflux.error.exceptions import StopImageGenerationException

PMSET_AC_POWER_STATUS = "Now drawing from 'AC Power'"
PMSET_BATT_STATUS_PATTERN = r"InternalBattery-.+?(\d+)%"

logger = logging.getLogger(__name__)


def _get_machine_model() -> str:
    """Get the Mac machine model using system_profiler."""
    try:
        result = subprocess.run(
            ["system_profiler", "-json", "SPHardwareDataType"], capture_output=True, text=True, check=True
        )
        data = json.loads(result.stdout)
        return data["SPHardwareDataType"][0]["machine_model"]
    except (subprocess.CalledProcessError, json.JSONDecodeError, IndexError, KeyError) as e:
        logger.warning(f"Cannot determine machine model via 'system_profiler -json SPHardwareDataType': {e}")
        return "Unknown"


MACHINE_MODEL = _get_machine_model()
# assumption: all Apple Silicon models powered by battery are "MacBook"s
MACHINE_IS_BATTERY_POWERED = "MacBook" in MACHINE_MODEL


def get_battery_percentage() -> int | None:
    """Get the current battery percentage of a battery-powered Mac.
    Returns None if Mac is not a battery-powered machine."""
    if not MACHINE_IS_BATTERY_POWERED:
        return None
    percentage = None
    try:
        # running the subprocess would be expensive in a tight loop
        # but in mflux use case, we would call this only once every
        # few minutes due to N-minutes-long generation times
        result = subprocess.run(["pmset", "-g", "batt"], capture_output=True, text=True, check=True)
        if PMSET_AC_POWER_STATUS not in result.stdout:
            if match := re.search(PMSET_BATT_STATUS_PATTERN, result.stdout):
                percentage = int(match.group(1))
    except (subprocess.CalledProcessError, TypeError) as e:
        logger.warning(
            f"Cannot read battery percentage via 'pmset -g batt': {e}. Battery saver functionality is disabled and the program will continue running."
        )

    return percentage


class BatterySaver(BeforeLoopCallback):
    def __init__(self, battery_percentage_stop_limit=10):
        self.limit = battery_percentage_stop_limit

    def call_before_loop(self, **kwargs) -> None:  # type: ignore
        current_pct: int | None = get_battery_percentage()
        if current_pct is not None and current_pct <= self.limit:
            raise StopImageGenerationException(f"Battery below {self.limit}% threshold: {current_pct}%")
