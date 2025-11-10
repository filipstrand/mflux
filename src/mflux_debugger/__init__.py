"""
Debugger for PyTorch and MLX.

Provides interactive debugging capabilities optimized for ML workloads with:
- Selective tracing (skips ML library internals)
- MLX lazy evaluation support
- Async execution for long-running operations
- Rich context capture (code, call stack, variables)
- Automatic checkpoint logging (per-script JSON files)
- Multiple transport layers (MCP, FastAPI)
"""

__version__ = "0.3.0"

from mflux_debugger.debugger import Debugger
from mflux_debugger.debugger_service import DebuggerService, get_debugger_service

__all__ = [
    "Debugger",
    "DebuggerService",
    "get_debugger_service",
]
