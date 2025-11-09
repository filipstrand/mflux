"""
Coverage report generator for dead code detection.

Generates marked-up file copies showing which lines were executed.
"""

import ast
from pathlib import Path
from typing import Set


def generate_marked_up_file(file_path: str, executed_lines: Set[int], output_path: Path) -> None:
    """
    Generate a marked-up copy of a file showing which lines were executed.

    Super simple approach:
    - ✅ (green) = line was executed
    - ❌ (red) = line exists but wasn't executed (dead code)
    - ⚪ (white) = line is not executable (blank, comment, function parameters, etc.)

    Args:
        file_path: Path to source file
        executed_lines: Set of executed line numbers
        output_path: Path where marked-up file should be saved
    """
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            source = f.read()
            lines = source.splitlines(keepends=True)
    except Exception:  # noqa: BLE001
        return

    # Parse AST to identify function parameter lines
    # Parameter lines are part of function signatures but not executed by Python's trace
    parameter_lines = set()
    try:
        tree = ast.parse(source, filename=file_path)

        class ParameterLineVisitor(ast.NodeVisitor):
            def __init__(self):
                self.param_lines = set()

            def _find_first_executable_line(self, body):
                """Find the first executable line in function body (skip comments/docstrings)."""
                for stmt in body:
                    if isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Constant):
                        # Skip docstrings
                        if isinstance(stmt.value.value, str):
                            continue
                    # Found first executable statement
                    return stmt.lineno
                return None

            def visit_FunctionDef(self, node):
                # Find first executable line in body (skip docstrings/comments)
                first_executable_line = self._find_first_executable_line(node.body) if node.body else None

                # If function body was executed, mark parameter lines as non-executable (not dead)
                if first_executable_line and first_executable_line in executed_lines:
                    # Mark all lines from function def to first executable body line as parameter lines
                    def_line = node.lineno
                    for line_num in range(def_line + 1, first_executable_line):
                        self.param_lines.add(line_num)
                self.generic_visit(node)

            def visit_AsyncFunctionDef(self, node):
                # Same logic for async functions
                first_executable_line = self._find_first_executable_line(node.body) if node.body else None
                if first_executable_line and first_executable_line in executed_lines:
                    def_line = node.lineno
                    for line_num in range(def_line + 1, first_executable_line):
                        self.param_lines.add(line_num)
                self.generic_visit(node)

        visitor = ParameterLineVisitor()
        visitor.visit(tree)
        parameter_lines = visitor.param_lines
    except Exception:  # noqa: BLE001
        # If AST parsing fails, just continue without parameter line detection
        pass

    # Create output directory if needed
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        for line_num, line_content in enumerate(lines, start=1):
            # Determine marker based on line type and execution status
            stripped = line_content.rstrip()

            # Check if line is executable (not blank, not just a comment)
            is_executable = bool(stripped) and not stripped.strip().startswith("#")

            if is_executable:
                # Check if this is a parameter line (part of function signature)
                if line_num in parameter_lines:
                    # Parameter lines are non-executable (part of signature, not dead code)
                    marker = "⚪"
                elif line_num in executed_lines:
                    marker = "✅"  # Hit
                else:
                    marker = "❌"  # Not hit (dead code)
            else:
                # Line is not executable (blank or comment)
                marker = "⚪"  # Not in scope

            # Write marked-up line: marker + space + line number + " | " + content
            f.write(f"{marker} {line_num:4d} | {line_content}")
