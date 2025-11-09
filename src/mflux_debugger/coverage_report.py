"""
Coverage report generator for dead code detection.

Analyzes coverage data to identify:
- Lines never executed
- Branches never taken
- Dead code paths
"""

import ast
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple


@dataclass
class BranchInfo:
    """Information about a branch (if/else/elif)."""

    file: str
    line: int
    branch_type: str  # "if", "elif", "else"
    condition: str  # The condition expression
    executed: bool  # Whether this branch was taken
    else_line: Optional[int] = None  # Line number of else block (if exists)


@dataclass
class UnusedDefinition:
    """Information about an unused class or function definition."""

    file: str
    line: int
    name: str
    type: str  # "class", "function", or "method"
    definition_line: int  # Line where class/function is defined


@dataclass
class CoverageReport:
    """Coverage analysis report."""

    script_path: str
    executed_lines: Dict[str, Set[int]]  # file -> set of executed line numbers
    total_lines: Dict[str, int]  # file -> total executable lines
    coverage_percentage: Dict[str, float]  # file -> coverage percentage
    dead_lines: Dict[str, List[int]]  # file -> list of unexecuted lines
    branches: List[BranchInfo]  # All branches found
    dead_branches: List[BranchInfo]  # Branches never taken
    unused_definitions: List[UnusedDefinition]  # Classes/functions defined but never used


class CoverageAnalyzer:
    """Analyzes coverage data to find dead code."""

    def __init__(self, coverage_data: Dict[str, Set[int]]):
        """
        Initialize analyzer with coverage data.

        Args:
            coverage_data: Dictionary mapping file paths to sets of executed line numbers
        """
        self.coverage_data = coverage_data

    def analyze_file(self, file_path: str) -> Tuple[List[int], List[BranchInfo], List[UnusedDefinition]]:
        """
        Analyze a single file for dead code.

        Args:
            file_path: Path to Python file

        Returns:
            Tuple of (dead_lines, branches, unused_definitions)
        """
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                source = f.read()
        except Exception:  # noqa: BLE001
            return [], [], []

        # Parse AST to find executable lines and branches
        try:
            tree = ast.parse(source, filename=file_path)
        except SyntaxError:
            return [], [], []

        # Find all executable lines
        executable_lines = self._find_executable_lines(tree, source)

        # Find executed lines for this file
        executed_lines = self.coverage_data.get(file_path, set())

        # Find dead lines
        dead_lines = [line for line in executable_lines if line not in executed_lines]

        # Find branches
        branches = self._find_branches(tree, file_path, executed_lines)

        # Find unused classes and functions
        unused_definitions = self._find_unused_definitions(tree, file_path, executed_lines)

        return dead_lines, branches, unused_definitions

    def _find_executable_lines(self, tree: ast.AST, source: str) -> List[int]:
        """Find all executable line numbers in the AST."""
        executable_lines = set()

        for node in ast.walk(tree):
            if hasattr(node, "lineno") and node.lineno:
                # Skip docstrings and comments
                if isinstance(node, (ast.Expr, ast.Constant)):
                    # Check if it's a docstring
                    if isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant):
                        if isinstance(node.value.value, str):
                            continue

                executable_lines.add(node.lineno)

        return sorted(executable_lines)

    def _find_branches(self, tree: ast.AST, file_path: str, executed_lines: Set[int]) -> List[BranchInfo]:
        """Find all if/elif/else branches in the AST."""

        class BranchVisitor(ast.NodeVisitor):
            def __init__(self, file_path: str, executed_lines: Set[int]):
                self.file_path = file_path
                self.executed_lines = executed_lines
                self.branches = []

            def visit_If(self, node: ast.If):
                # Track the if branch
                if_line = node.lineno
                # Try to get condition as string (fallback to repr if unparse not available)
                try:
                    if hasattr(ast, "unparse"):
                        condition = ast.unparse(node.test)
                    else:
                        # Fallback: use repr for older Python versions
                        condition = repr(node.test)
                except Exception:  # noqa: BLE001
                    condition = str(node.test)

                # Check if if branch was executed (check first line of if body)
                if_body_executed = False
                if node.body and node.body[0].lineno in self.executed_lines:
                    if_body_executed = True

                # Find else line if exists
                else_line = None
                if node.orelse:
                    # Check if it's an elif (orelse contains another If)
                    if len(node.orelse) == 1 and isinstance(node.orelse[0], ast.If):
                        # This is an elif - recurse
                        self.visit_If(node.orelse[0])
                    else:
                        # This is an else
                        else_line = node.orelse[0].lineno if node.orelse else None

                self.branches.append(
                    BranchInfo(
                        file=self.file_path,
                        line=if_line,
                        branch_type="if",
                        condition=condition,
                        executed=if_body_executed,
                        else_line=else_line,
                    )
                )

                # Check if else branch was executed
                if else_line and else_line in self.executed_lines:
                    self.branches.append(
                        BranchInfo(
                            file=self.file_path,
                            line=else_line,
                            branch_type="else",
                            condition="else",
                            executed=True,
                        )
                    )
                elif else_line:
                    # Else branch exists but wasn't executed
                    self.branches.append(
                        BranchInfo(
                            file=self.file_path,
                            line=else_line,
                            branch_type="else",
                            condition="else",
                            executed=False,
                        )
                    )

                self.generic_visit(node)

        visitor = BranchVisitor(file_path, executed_lines)
        visitor.visit(tree)

        return visitor.branches

    def _find_unused_definitions(
        self, tree: ast.AST, file_path: str, executed_lines: Set[int]
    ) -> List[UnusedDefinition]:
        """Find classes and functions that are defined but never used (only definition line executed)."""

        class DefinitionVisitor(ast.NodeVisitor):
            def __init__(self, file_path: str, executed_lines: Set[int]):
                self.file_path = file_path
                self.executed_lines = executed_lines
                self.unused = []
                self.current_class = None  # Track if we're inside a class

            def _get_body_lines(self, node: ast.AST, exclude_definitions: bool = False) -> Set[int]:
                """Get all executable line numbers in a node's body.

                Args:
                    node: AST node with a body
                    exclude_definitions: If True, exclude function/class definition lines (only check their bodies)
                """
                body_lines = set()
                if hasattr(node, "body"):
                    for stmt in node.body:
                        # Skip function/class definitions if exclude_definitions is True
                        # (we only want to check if their bodies are executed, not the definition lines themselves)
                        if exclude_definitions and isinstance(
                            stmt, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
                        ):
                            # Recursively get body lines from the function/class, excluding its own definition
                            body_lines.update(self._get_body_lines(stmt, exclude_definitions=True))
                            continue

                        for child in ast.walk(stmt):
                            if hasattr(child, "lineno") and child.lineno:
                                # Skip docstrings
                                if isinstance(child, (ast.Expr, ast.Constant)):
                                    if isinstance(child, ast.Expr) and isinstance(child.value, ast.Constant):
                                        if isinstance(child.value.value, str):
                                            continue
                                # Skip the definition line itself if exclude_definitions is True
                                if exclude_definitions and isinstance(
                                    stmt, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
                                ):
                                    if child.lineno == stmt.lineno:
                                        continue
                                body_lines.add(child.lineno)
                return body_lines

            def visit_ClassDef(self, node: ast.ClassDef):
                # Check if class definition line is executed
                if node.lineno not in self.executed_lines:
                    self.generic_visit(node)
                    return

                # Save current class context
                old_class = self.current_class
                self.current_class = node

                # Get all body lines (methods, assignments, etc.)
                # Exclude method definition lines - we only care if method bodies are executed
                body_lines = self._get_body_lines(node, exclude_definitions=True)

                # Check if any body line is executed
                body_executed = any(line in self.executed_lines for line in body_lines)

                if not body_executed:
                    # Class is defined but never instantiated or called
                    self.unused.append(
                        UnusedDefinition(
                            file=self.file_path,
                            line=node.lineno,
                            name=node.name,
                            type="class",
                            definition_line=node.lineno,
                        )
                    )

                self.generic_visit(node)

                # Restore class context
                self.current_class = old_class

            def visit_FunctionDef(self, node: ast.FunctionDef):
                # Check methods inside classes for unused methods
                if self.current_class is not None:
                    # Check if method definition line is executed
                    if node.lineno in self.executed_lines:
                        # Get all body lines (excluding nested function definitions)
                        body_lines = self._get_body_lines(node, exclude_definitions=True)

                        # Check if any body line is executed
                        body_executed = any(line in self.executed_lines for line in body_lines)

                        if not body_executed:
                            # Method is defined but never called
                            self.unused.append(
                                UnusedDefinition(
                                    file=self.file_path,
                                    line=node.lineno,
                                    name=node.name,
                                    type="method",
                                    definition_line=node.lineno,
                                )
                            )
                    self.generic_visit(node)
                    return

                # Check if function definition line is executed
                if node.lineno not in self.executed_lines:
                    self.generic_visit(node)
                    return

                # Get all body lines (excluding nested function definitions)
                body_lines = self._get_body_lines(node, exclude_definitions=True)

                # Check if any body line is executed
                body_executed = any(line in self.executed_lines for line in body_lines)

                if not body_executed:
                    # Function is defined but never called
                    self.unused.append(
                        UnusedDefinition(
                            file=self.file_path,
                            line=node.lineno,
                            name=node.name,
                            type="function",
                            definition_line=node.lineno,
                        )
                    )

                self.generic_visit(node)

            def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef):
                # Check methods inside classes for unused methods
                if self.current_class is not None:
                    # Check if method definition line is executed
                    if node.lineno in self.executed_lines:
                        # Get all body lines (excluding nested function definitions)
                        body_lines = self._get_body_lines(node, exclude_definitions=True)

                        # Check if any body line is executed
                        body_executed = any(line in self.executed_lines for line in body_lines)

                        if not body_executed:
                            # Method is defined but never called
                            self.unused.append(
                                UnusedDefinition(
                                    file=self.file_path,
                                    line=node.lineno,
                                    name=node.name,
                                    type="method",
                                    definition_line=node.lineno,
                                )
                            )
                    self.generic_visit(node)
                    return

                # Same logic as FunctionDef
                if node.lineno not in self.executed_lines:
                    self.generic_visit(node)
                    return

                body_lines = self._get_body_lines(node, exclude_definitions=True)
                body_executed = any(line in self.executed_lines for line in body_lines)

                if not body_executed:
                    self.unused.append(
                        UnusedDefinition(
                            file=self.file_path,
                            line=node.lineno,
                            name=node.name,
                            type="function",
                            definition_line=node.lineno,
                        )
                    )

                self.generic_visit(node)

        visitor = DefinitionVisitor(file_path, executed_lines)
        visitor.visit(tree)

        return visitor.unused

    def generate_report(self, script_path: str, watch_files: Optional[Set[str]] = None) -> CoverageReport:
        """
        Generate coverage report.

        Args:
            script_path: Path to the script that was executed
            watch_files: Optional set of files to analyze (if None, analyzes all files in coverage_data)

        Returns:
            CoverageReport with analysis results
        """
        files_to_analyze = watch_files if watch_files else set(self.coverage_data.keys())

        total_lines = {}
        dead_lines = {}
        all_branches = []
        dead_branches = []
        all_unused_definitions = []

        for file_path in files_to_analyze:
            file_dead_lines, file_branches, file_unused = self.analyze_file(file_path)

            # Count total executable lines
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    lines = f.readlines()
                total_lines[file_path] = len(
                    [line for line in lines if line.strip() and not line.strip().startswith("#")]
                )
            except Exception:  # noqa: BLE001
                total_lines[file_path] = 0

            dead_lines[file_path] = file_dead_lines

            # Collect branches
            for branch in file_branches:
                all_branches.append(branch)
                if not branch.executed:
                    dead_branches.append(branch)

            # Collect unused definitions
            all_unused_definitions.extend(file_unused)

        # Calculate coverage percentages
        coverage_percentage = {}
        for file_path in files_to_analyze:
            executed_count = len(self.coverage_data.get(file_path, set()))
            total = total_lines.get(file_path, 0)
            if total > 0:
                coverage_percentage[file_path] = (executed_count / total) * 100
            else:
                coverage_percentage[file_path] = 0.0

        return CoverageReport(
            script_path=script_path,
            executed_lines=self.coverage_data,
            total_lines=total_lines,
            coverage_percentage=coverage_percentage,
            dead_lines=dead_lines,
            branches=all_branches,
            dead_branches=dead_branches,
            unused_definitions=all_unused_definitions,
        )


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
