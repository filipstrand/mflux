"""
Lightweight ML Debugger for PyTorch and MLX.

Provides interactive debugging capabilities optimized for ML workloads with:
- Selective tracing (skips ML library internals)
- MLX lazy evaluation support
- Async execution for long-running operations
- Rich context capture (code, call stack, variables)
- Trace recording for offline analysis
- Multiple transport layers (MCP, FastAPI)
"""

__version__ = "0.2.0"

from mflux_debugger.debugger_service import DebuggerService, get_debugger_service
from mflux_debugger.lightweight_debugger import LightweightDebugger
from mflux_debugger.trace_recorder import TraceRecorder

__all__ = [
    "LightweightDebugger",
    "DebuggerService",
    "get_debugger_service",
    "TraceRecorder",
]
