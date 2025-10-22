"""
MCP server adapter for lightweight ML debugger.

This is a thin transport layer that exposes the debugger service via MCP.
All business logic lives in debugger_service.py.
"""

import asyncio
import logging

from mcp.server import Server
from mcp.types import TextContent, Tool

from mflux_debugger.debugger_service import get_debugger_service

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


async def main():
    """Run the MCP server."""
    server = Server("pytorch-debugger-lightweight")

    # Get the debugger service singleton
    service = get_debugger_service()

    @server.list_tools()
    async def list_tools() -> list[Tool]:
        """List available debugging tools."""
        return [
            Tool(
                name="debug_start_session",
                description="Start a debugging session for a Python script",
                inputSchema={
                    "type": "object",
                    "properties": {"script_path": {"type": "string", "description": "Path to the Python script"}},
                    "required": ["script_path"],
                },
            ),
            Tool(
                name="debug_set_breakpoint",
                description="Set a breakpoint at a specific line",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "file_path": {"type": "string", "description": "Path to the file"},
                        "line": {"type": "number", "description": "Line number"},
                        "condition": {"type": "string", "description": "Optional condition (Python expression)"},
                    },
                    "required": ["file_path", "line"],
                },
            ),
            Tool(
                name="debug_continue",
                description="Continue execution until next breakpoint",
                inputSchema={"type": "object", "properties": {}},
            ),
            Tool(
                name="debug_step_over",
                description="Step over the current line",
                inputSchema={"type": "object", "properties": {}},
            ),
            Tool(
                name="debug_step_into",
                description="Step into a function call",
                inputSchema={"type": "object", "properties": {}},
            ),
            Tool(
                name="debug_step_out",
                description="Step out of the current function",
                inputSchema={"type": "object", "properties": {}},
            ),
            Tool(
                name="debug_list_variables",
                description="List all variables in the current scope",
                inputSchema={"type": "object", "properties": {}},
            ),
            Tool(
                name="debug_inspect_variable",
                description="Inspect a specific variable with optional statistics",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "name": {"type": "string", "description": "Variable name"},
                        "show_stats": {
                            "type": "boolean",
                            "description": "Show statistics for tensors",
                            "default": False,
                        },
                    },
                    "required": ["name"],
                },
            ),
            Tool(
                name="debug_evaluate",
                description="Evaluate a Python expression in the current context",
                inputSchema={
                    "type": "object",
                    "properties": {"expression": {"type": "string", "description": "Python expression to evaluate"}},
                    "required": ["expression"],
                },
            ),
            Tool(
                name="debug_remove_breakpoint",
                description="Remove a breakpoint at a specific location",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "file_path": {"type": "string", "description": "Path to the file"},
                        "line": {"type": "integer", "description": "Line number"},
                    },
                    "required": ["file_path", "line"],
                },
            ),
            Tool(
                name="debug_list_breakpoints",
                description="List all currently set breakpoints",
                inputSchema={"type": "object", "properties": {}},
            ),
            Tool(
                name="debug_clear_all_breakpoints",
                description="Remove all breakpoints",
                inputSchema={"type": "object", "properties": {}},
            ),
            Tool(
                name="debug_get_location",
                description="Get the current execution location",
                inputSchema={"type": "object", "properties": {}},
            ),
            Tool(
                name="debug_check_status",
                description="Check the current debugger status",
                inputSchema={"type": "object", "properties": {}},
            ),
            Tool(
                name="debug_terminate",
                description="Terminate the debugging session",
                inputSchema={"type": "object", "properties": {}},
            ),
        ]

    @server.call_tool()
    async def call_tool(name: str, arguments: dict) -> list[TextContent]:
        """Handle tool calls by delegating to the debugger service."""
        try:
            # Route to appropriate service method
            if name == "debug_start_session":
                response = service.start_session(arguments["script_path"])

            elif name == "debug_set_breakpoint":
                response = service.set_breakpoint(
                    file_path=arguments["file_path"], line=arguments["line"], condition=arguments.get("condition")
                )

            elif name == "debug_remove_breakpoint":
                response = service.remove_breakpoint(file_path=arguments["file_path"], line=arguments["line"])

            elif name == "debug_continue":
                response = service.continue_execution()

            elif name == "debug_step_over":
                response = service.step_over()

            elif name == "debug_step_into":
                response = service.step_into()

            elif name == "debug_step_out":
                response = service.step_out()

            elif name == "debug_list_variables":
                response = service.list_variables()

            elif name == "debug_inspect_variable":
                response = service.inspect_variable(
                    name=arguments["name"], show_stats=arguments.get("show_stats", False)
                )

            elif name == "debug_evaluate":
                response = service.evaluate(expression=arguments["expression"])

            elif name == "debug_list_breakpoints":
                response = service.list_breakpoints()

            elif name == "debug_clear_all_breakpoints":
                response = service.clear_all_breakpoints()

            elif name == "debug_get_location":
                response = service.get_location()

            elif name == "debug_check_status":
                response = service.check_status()

            elif name == "debug_terminate":
                response = service.terminate()

            else:
                return [TextContent(type="text", text=f"❌ Unknown tool: {name}")]

            # Convert service response to MCP TextContent
            return [TextContent(type="text", text=response.message)]

        except Exception as e:  # noqa: BLE001
            logger.error(f"Tool execution failed: {e}", exc_info=True)
            return [TextContent(type="text", text=f"❌ Error: {e}")]

    # Run the server
    from mcp.server.stdio import stdio_server

    async with stdio_server() as (read_stream, write_stream):
        logger.info("Lightweight ML debugger MCP server started")
        await server.run(read_stream, write_stream, server.create_initialization_options())


def cli_main():
    """Entry point for command-line usage."""
    asyncio.run(main())


if __name__ == "__main__":
    cli_main()
