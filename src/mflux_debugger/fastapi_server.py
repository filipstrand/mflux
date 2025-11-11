"""
FastAPI server adapter for debugger.

Example showing how to expose the same debugger service via REST API.
This demonstrates the transport-agnostic design - same service, different interface.

Usage:
    pip install fastapi uvicorn
    uvicorn mflux_debugger.fastapi_server:app --reload
"""

from difflib import get_close_matches

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from mflux_debugger import tensor_debug
from mflux_debugger.debugger_service import DebuggerResponse, get_debugger_service

app = FastAPI(title="ML Debugger API", description="REST API for lightweight ML debugging")

# Get the debugger service singleton
service = get_debugger_service()


# Request/Response models
class StartSessionRequest(BaseModel):
    script_path: str
    framework: str | None = None  # "pytorch" or "mlx" - optional, will auto-detect if not provided
    clear_tensors: bool | None = (
        None  # None = default behavior (clear for PyTorch, keep for MLX), True/False = override
    )
    coverage_mode: bool = False  # Enable coverage tracking for dead code detection


class SetBreakpointRequest(BaseModel):
    file_path: str
    line: int
    condition: str | None = None


class RemoveBreakpointRequest(BaseModel):
    file_path: str
    line: int


class EvaluateRequest(BaseModel):
    expression: str


class SetCheckpointBreakpointRequest(BaseModel):
    checkpoint_name: str
    description: str = ""


class SetCheckpointRecordAllRequest(BaseModel):
    enabled: bool


class DebugResponse(BaseModel):
    success: bool
    message: str
    data: dict | None = None
    error: str | None = None


def _make_response(response):
    """Helper to convert DebuggerResponse to consistent JSON response."""
    status_code = 200 if response.success else 400
    return JSONResponse(
        status_code=status_code,
        content={
            "success": response.success,
            "message": response.message,
            "data": response.data,
            "error": response.error,
        },
    )


# Endpoints
@app.post("/debug/start")
def start_session(request: StartSessionRequest):
    """Start a debugging session."""
    return _make_response(
        service.start_session(
            request.script_path,
            framework=request.framework,
            clear_tensors=request.clear_tensors,
            coverage_mode=request.coverage_mode,
        )
    )


@app.post("/debug/breakpoint")
def set_breakpoint(request: SetBreakpointRequest):
    """Set a breakpoint."""
    return _make_response(service.set_breakpoint(request.file_path, request.line, request.condition))


@app.post("/debug/breakpoint/remove")
def remove_breakpoint(request: RemoveBreakpointRequest):
    """Remove a breakpoint."""
    return _make_response(service.remove_breakpoint(request.file_path, request.line))


@app.get("/debug/breakpoints")
def list_breakpoints():
    """List all breakpoints."""
    return _make_response(service.list_breakpoints())


@app.post("/debug/continue")
def continue_execution():
    """Continue execution (blocking - waits for breakpoint or completion)."""
    return _make_response(service.continue_execution())


@app.post("/debug/continue_async")
def continue_execution_async():
    """
    Continue execution in background (non-blocking).

    Perfect for ML workloads with heavy model loading or long inference.
    Returns immediately - use /debug/status or /debug/location to poll for results.
    """
    return _make_response(service.continue_execution_async())


@app.get("/debug/variables")
def list_variables():
    """List all variables."""
    return _make_response(service.list_variables())


@app.post("/debug/evaluate")
def evaluate(request: EvaluateRequest):
    """Evaluate an expression."""
    return _make_response(service.evaluate(request.expression))


@app.get("/debug/location")
def get_location():
    """Get current location."""
    return _make_response(service.get_location())


@app.get("/debug/coverage")
def get_coverage():
    """Get coverage data (file -> list of executed line numbers)."""
    return _make_response(service.get_coverage())


@app.get("/debug/status", response_model=DebugResponse)
def check_status():
    """Check debugger status."""
    return service.check_status()


@app.post("/debug/terminate", response_model=DebugResponse)
def terminate():
    """Terminate debugging session."""
    return service.terminate()


# Semantic checkpoint endpoints
@app.post("/debug/checkpoint/break")
def set_checkpoint_breakpoint(request: SetCheckpointBreakpointRequest):
    """Set a semantic checkpoint breakpoint."""
    return _make_response(service.set_checkpoint_breakpoint(request.checkpoint_name, request.description))


@app.post("/debug/checkpoint/remove")
def remove_checkpoint_breakpoint(checkpoint_name: str):
    """Remove a semantic checkpoint breakpoint."""
    return _make_response(service.remove_checkpoint_breakpoint(checkpoint_name))


@app.post("/debug/checkpoint/record-all")
def set_checkpoint_record_all(request: SetCheckpointRecordAllRequest):
    """Enable or disable recording of ALL semantic checkpoints."""
    return _make_response(service.set_checkpoint_record_all(request.enabled))


@app.post("/debug/checkpoint/break-all")
def set_checkpoint_break_all(request: SetCheckpointRecordAllRequest):
    """Enable or disable breaking at ALL semantic checkpoints."""
    return _make_response(service.set_checkpoint_break_all(request.enabled))


@app.get("/debug/checkpoints")
def list_checkpoint_breakpoints():
    """List all checkpoint breakpoints."""
    return _make_response(service.list_checkpoint_breakpoints())


@app.get("/debug/checkpoint/current")
def get_current_checkpoint():
    """Get the current checkpoint if paused at one."""
    return _make_response(service.get_current_checkpoint())


@app.get("/debug/checkpoint/history")
def get_checkpoint_history():
    """Get history of all checkpoints hit in this session."""
    return _make_response(service.get_checkpoint_history())


@app.get("/debug/checkpoint/verification")
def get_checkpoint_verification():
    """Get checkpoint verification status with hit counts and execution order."""
    return _make_response(service.get_checkpoint_verification_status())


@app.post("/debug/checkpoint/break")
def set_conditional_checkpoint_breakpoint(request: dict):
    """
    Set a conditional breakpoint on a checkpoint with context matching.

    Request body:
    {
        "checkpoint_name": "sdpa_05_output",
        "condition": {"block": 0, "timestep": 0, "hit_count": 1}
    }
    """
    checkpoint_name = request.get("checkpoint_name")
    condition = request.get("condition", {})

    if not checkpoint_name:
        return _make_response(
            DebuggerResponse(success=False, message="checkpoint_name is required", error="Missing checkpoint_name")
        )

    return _make_response(service.set_conditional_checkpoint_breakpoint(checkpoint_name, condition))


@app.get("/")
def root():
    """Root endpoint with quick start guide and common endpoints."""
    return {
        "name": "ML Debugger API",
        "version": "1.0",
        "message": "Lightweight debugging for ML workflows",
        "documentation": {
            "interactive_docs": "/docs",
            "openapi_spec": "/openapi.json",
        },
        "quick_start": {
            "1_start_session": "POST /debug/start",
            "2_set_breakpoints": "POST /debug/breakpoint",
            "3_run": "POST /debug/continue or POST /debug/continue_async",
            "4_inspect": "GET /debug/variables, POST /debug/inspect, POST /debug/evaluate",
            "5_terminate": "POST /debug/terminate",
        },
        "common_endpoints": {
            "session": [
                "POST /debug/start - Start debugging session",
                "GET /debug/status - Check current status",
                "POST /debug/terminate - End session and save trace",
            ],
            "breakpoints": [
                "POST /debug/breakpoint - Set a breakpoint",
                "GET /debug/breakpoints - List all breakpoints",
                "POST /debug/breakpoint/remove - Remove specific breakpoint",
            ],
            "execution": [
                "POST /debug/continue - Run until next breakpoint (blocking)",
                "POST /debug/continue_async - Run in background (non-blocking)",
            ],
            "inspection": [
                "GET /debug/variables - List all variables in scope",
                "POST /debug/inspect - Inspect specific variable with stats",
                "POST /debug/inspect_tensor - Smart tensor inspection (MLX/PyTorch/NumPy)",
                "POST /debug/evaluate - Evaluate arbitrary expression",
                "GET /debug/location - Get current execution location",
            ],
        },
    }


# Custom 404 handler with smart suggestions
# Debug Tensors Endpoints
class DebugTensorsClearRequest(BaseModel):
    name: str | None = None
    confirm: bool = True


@app.get("/debug_tensors/list")
def debug_tensors_list_endpoint():
    """List all available debug tensors."""
    try:
        tensors = tensor_debug.debug_list()
        return JSONResponse(
            status_code=200,
            content={
                "success": True,
                "data": {
                    "tensors": tensors,
                    "count": len(tensors),
                },
                "message": f"Found {len(tensors)} debug tensor(s)",
            },
        )
    except Exception as e:  # noqa: BLE001
        return JSONResponse(
            status_code=500,
            content={"success": False, "error": str(e), "message": f"Failed to list debug tensors: {e}"},
        )


@app.get("/debug_tensors/info/{name}")
def debug_tensors_info_endpoint(name: str):
    """Get information about a specific debug tensor."""
    try:
        info = tensor_debug.debug_info(name)
        return JSONResponse(
            status_code=200, content={"success": True, "data": info, "message": f"Tensor info for '{name}'"}
        )
    except FileNotFoundError as e:
        return JSONResponse(
            status_code=404, content={"success": False, "error": str(e), "message": f"Tensor '{name}' not found"}
        )
    except Exception as e:  # noqa: BLE001
        return JSONResponse(
            status_code=500, content={"success": False, "error": str(e), "message": f"Failed to get tensor info: {e}"}
        )


@app.post("/debug_tensors/clear")
def debug_tensors_clear_endpoint(request: DebugTensorsClearRequest):
    """Clear debug tensors from disk."""
    try:
        deleted = tensor_debug.debug_clear(name=request.name, confirm=False)  # API doesn't confirm
        return JSONResponse(
            status_code=200,
            content={"success": True, "data": {"deleted": deleted}, "message": f"Deleted {deleted} file(s)"},
        )
    except Exception as e:  # noqa: BLE001
        return JSONResponse(
            status_code=500, content={"success": False, "error": str(e), "message": f"Failed to clear tensors: {e}"}
        )


@app.exception_handler(404)
async def custom_404_handler(request: Request, exc: HTTPException):
    """Smart 404 handler that suggests similar endpoints."""
    requested_path = request.url.path

    # Get all registered routes
    all_routes = [route.path for route in app.routes if hasattr(route, "path")]

    # Find close matches
    suggestions = get_close_matches(requested_path, all_routes, n=3, cutoff=0.6)

    response = {
        "error": "Not Found",
        "message": f"Endpoint '{requested_path}' does not exist",
        "requested": requested_path,
    }

    if suggestions:
        response["did_you_mean"] = suggestions
        response["tip"] = f"Try: {suggestions[0]}"
    else:
        response["tip"] = "Visit / for a list of all endpoints, or /docs for interactive documentation"

    return JSONResponse(
        status_code=404,
        content=response,
    )


# CLI Support for custom port configuration
if __name__ == "__main__":
    import argparse

    import uvicorn

    parser = argparse.ArgumentParser(description="ML Debugger API Server")
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port to run the server on (default: 8000)",
    )
    parser.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
        help="Host to bind to (default: 127.0.0.1)",
    )
    parser.add_argument(
        "--reload",
        action="store_true",
        help="Enable auto-reload for development",
    )

    args = parser.parse_args()

    print(f"🚀 Starting ML Debugger API on {args.host}:{args.port}")
    if args.reload:
        print("🔄 Auto-reload enabled")

    uvicorn.run(
        "mflux_debugger.fastapi_server:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
    )
