"""
FastAPI server adapter for lightweight ML debugger.

Example showing how to expose the same debugger service via REST API.
This demonstrates the transport-agnostic design - same service, different interface.

Usage:
    pip install fastapi uvicorn
    uvicorn mflux_debugger.fastapi_server:app --reload
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from mflux_debugger.debugger_service import get_debugger_service

app = FastAPI(title="ML Debugger API", description="REST API for lightweight ML debugging")

# Get the debugger service singleton
service = get_debugger_service()


# Request/Response models
class StartSessionRequest(BaseModel):
    script_path: str


class SetBreakpointRequest(BaseModel):
    file_path: str
    line: int
    condition: str | None = None


class RemoveBreakpointRequest(BaseModel):
    file_path: str
    line: int


class InspectVariableRequest(BaseModel):
    name: str
    show_stats: bool = False


class EvaluateRequest(BaseModel):
    expression: str


class DebugResponse(BaseModel):
    success: bool
    message: str
    data: dict | None = None
    error: str | None = None


# Endpoints
@app.post("/debug/start", response_model=DebugResponse)
def start_session(request: StartSessionRequest):
    """Start a debugging session."""
    response = service.start_session(request.script_path)
    if not response.success:
        raise HTTPException(status_code=400, detail=response.error)
    return response


@app.post("/debug/breakpoint", response_model=DebugResponse)
def set_breakpoint(request: SetBreakpointRequest):
    """Set a breakpoint."""
    response = service.set_breakpoint(request.file_path, request.line, request.condition)
    if not response.success:
        raise HTTPException(status_code=400, detail=response.error)
    return response


@app.post("/debug/breakpoint/remove", response_model=DebugResponse)
def remove_breakpoint(request: RemoveBreakpointRequest):
    """Remove a breakpoint."""
    response = service.remove_breakpoint(request.file_path, request.line)
    if not response.success:
        raise HTTPException(status_code=400, detail=response.error)
    return response


@app.get("/debug/breakpoints", response_model=DebugResponse)
def list_breakpoints():
    """List all breakpoints."""
    response = service.list_breakpoints()
    if not response.success:
        raise HTTPException(status_code=400, detail=response.error)
    return response


@app.post("/debug/breakpoints/clear", response_model=DebugResponse)
def clear_all_breakpoints():
    """Clear all breakpoints."""
    response = service.clear_all_breakpoints()
    if not response.success:
        raise HTTPException(status_code=400, detail=response.error)
    return response


@app.post("/debug/continue", response_model=DebugResponse)
def continue_execution():
    """Continue execution (blocking - waits for breakpoint or completion)."""
    response = service.continue_execution()
    if not response.success:
        raise HTTPException(status_code=400, detail=response.error)
    return response


@app.post("/debug/continue_async", response_model=DebugResponse)
def continue_execution_async():
    """
    Continue execution in background (non-blocking).

    Perfect for ML workloads with heavy model loading or long inference.
    Returns immediately - use /debug/status or /debug/location to poll for results.
    """
    response = service.continue_execution_async()
    if not response.success:
        raise HTTPException(status_code=400, detail=response.error)
    return response


@app.post("/debug/step/over", response_model=DebugResponse)
def step_over():
    """Step over current line."""
    response = service.step_over()
    if not response.success:
        raise HTTPException(status_code=400, detail=response.error)
    return response


@app.post("/debug/step/into", response_model=DebugResponse)
def step_into():
    """Step into function."""
    response = service.step_into()
    if not response.success:
        raise HTTPException(status_code=400, detail=response.error)
    return response


@app.post("/debug/step/out", response_model=DebugResponse)
def step_out():
    """Step out of function."""
    response = service.step_out()
    if not response.success:
        raise HTTPException(status_code=400, detail=response.error)
    return response


@app.get("/debug/variables", response_model=DebugResponse)
def list_variables():
    """List all variables."""
    response = service.list_variables()
    if not response.success:
        raise HTTPException(status_code=400, detail=response.error)
    return response


@app.post("/debug/inspect", response_model=DebugResponse)
def inspect_variable(request: InspectVariableRequest):
    """Inspect a specific variable."""
    response = service.inspect_variable(request.name, request.show_stats)
    if not response.success:
        raise HTTPException(status_code=400, detail=response.error)
    return response


@app.post("/debug/evaluate", response_model=DebugResponse)
def evaluate(request: EvaluateRequest):
    """Evaluate an expression."""
    response = service.evaluate(request.expression)
    if not response.success:
        raise HTTPException(status_code=400, detail=response.error)
    return response


@app.get("/debug/location", response_model=DebugResponse)
def get_location():
    """Get current location."""
    response = service.get_location()
    if not response.success:
        raise HTTPException(status_code=400, detail=response.error)
    return response


@app.get("/debug/status", response_model=DebugResponse)
def check_status():
    """Check debugger status."""
    return service.check_status()


@app.post("/debug/terminate", response_model=DebugResponse)
def terminate():
    """Terminate debugging session."""
    return service.terminate()


@app.get("/")
def root():
    """Root endpoint."""
    return {
        "name": "ML Debugger API",
        "version": "1.0",
        "docs": "/docs",
        "message": "Lightweight debugging for ML workflows",
    }
