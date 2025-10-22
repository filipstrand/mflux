"""
Tee writer that writes to multiple outputs simultaneously.

Used to capture script stdout/stderr to both in-memory buffer and file.
"""

from typing import Any


class TeeWriter:
    """Write to multiple outputs simultaneously (like Unix tee command)."""

    def __init__(self, *outputs, prefix: str = ""):
        """
        Initialize tee writer.

        Args:
            *outputs: File-like objects to write to
            prefix: Optional prefix to add to each line
        """
        self.outputs = outputs
        self.prefix = prefix

    def write(self, text: str) -> int:
        """Write text to all outputs."""
        if self.prefix and text and text != "\n":
            text = self.prefix + text

        bytes_written = 0
        for output in self.outputs:
            try:
                bytes_written = output.write(text)
                # Flush immediately for real-time logging
                if hasattr(output, "flush"):
                    output.flush()
            except Exception:  # noqa: BLE001, PERF203, S110
                pass  # Continue even if one output fails

        return bytes_written

    def flush(self):
        """Flush all outputs."""
        for output in self.outputs:
            try:
                if hasattr(output, "flush"):
                    output.flush()
            except Exception:  # noqa: BLE001, PERF203, S110
                pass

    def isatty(self) -> bool:
        """Check if any output is a tty."""
        return any(hasattr(output, "isatty") and output.isatty() for output in self.outputs)

    def __getattr__(self, name: str) -> Any:
        """Forward other attributes to first output."""
        if self.outputs:
            return getattr(self.outputs[0], name)
        raise AttributeError(f"TeeWriter has no attribute '{name}'")
