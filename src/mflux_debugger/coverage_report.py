"""
Coverage report generator for dead code detection.

Analyzes coverage data to identify:
- Lines never executed
- Branches never taken
- Dead code paths
"""

import ast
from dataclasses import dataclass
from datetime import datetime
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


def _find_contiguous_blocks(dead_lines: List[int], max_gap: int = 2) -> List[Tuple[int, int]]:
    """Find contiguous blocks of dead lines (allowing small gaps)."""
    if not dead_lines:
        return []

    blocks = []
    start = dead_lines[0]
    end = dead_lines[0]

    for i in range(1, len(dead_lines)):
        if dead_lines[i] - end <= max_gap + 1:
            # Continue the block (allowing small gaps)
            end = dead_lines[i]
        else:
            # End current block, start new one
            blocks.append((start, end))
            start = dead_lines[i]
            end = dead_lines[i]

    blocks.append((start, end))
    return blocks


def _merge_nearby_blocks(blocks: List[Tuple[int, int]], max_distance: int = 30) -> List[Tuple[int, int]]:
    """Merge blocks that are close together (within max_distance lines)."""
    if not blocks:
        return []

    merged = []
    current_start, current_end = blocks[0]

    for block_start, block_end in blocks[1:]:
        # If the next block is close to the current one, merge them
        if block_start - current_end <= max_distance:
            # Merge: extend current block to include the next one
            current_end = max(current_end, block_end)
        else:
            # Too far apart, start a new block
            merged.append((current_start, current_end))
            current_start, current_end = block_start, block_end

    merged.append((current_start, current_end))
    return merged


def _get_context_lines(
    file_path: str, dead_line: int, executed_lines: Set[int], context_size: int = 3
) -> Tuple[List[Tuple[int, str, bool]], List[Tuple[int, str, bool]], List[Tuple[int, str, bool]]]:
    """Get context lines before, at, and after a dead line."""
    try:
        with open(file_path, "r", encoding="utf-8") as src:
            all_lines = src.readlines()
    except Exception:  # noqa: BLE001
        return [], [], []

    before_lines = []
    at_lines = []
    after_lines = []

    # Get lines before
    for i in range(max(0, dead_line - context_size - 1), dead_line - 1):
        if i < len(all_lines):
            is_executed = (i + 1) in executed_lines
            before_lines.append((i + 1, all_lines[i].rstrip(), is_executed))

    # Get the dead line itself
    if dead_line - 1 < len(all_lines):
        at_lines.append((dead_line, all_lines[dead_line - 1].rstrip(), False))

    # Get lines after
    for i in range(dead_line, min(len(all_lines), dead_line + context_size)):
        is_executed = (i + 1) in executed_lines
        after_lines.append((i + 1, all_lines[i].rstrip(), is_executed))

    return before_lines, at_lines, after_lines


def _find_unused_definition_body_ranges(
    tree: ast.AST, unused_definitions: List[UnusedDefinition]
) -> List[Tuple[int, int]]:
    """Find body line ranges for unused definitions (from definition line to end of body)."""
    # Separate by type - we need to handle classes, functions, and methods differently
    unused_class_lines = {unused.definition_line for unused in unused_definitions if unused.type == "class"}
    unused_function_lines = {unused.definition_line for unused in unused_definitions if unused.type == "function"}
    unused_method_lines = {unused.definition_line for unused in unused_definitions if unused.type == "method"}

    class BodyRangeVisitor(ast.NodeVisitor):
        def __init__(
            self, unused_class_lines: Set[int], unused_function_lines: Set[int], unused_method_lines: Set[int]
        ):
            self.unused_class_lines = unused_class_lines
            self.unused_function_lines = unused_function_lines
            self.unused_method_lines = unused_method_lines
            self.body_ranges = []
            self.current_class_line = None

        def visit_ClassDef(self, node: ast.ClassDef):
            old_class_line = self.current_class_line
            if node.lineno in self.unused_class_lines:
                self.current_class_line = node.lineno
                # Find the end of the class body
                if node.body:
                    end_line = node.lineno
                    for stmt in node.body:
                        for child in ast.walk(stmt):
                            if hasattr(child, "end_lineno") and child.end_lineno:
                                end_line = max(end_line, child.end_lineno)
                            elif hasattr(child, "lineno") and child.lineno:
                                end_line = max(end_line, child.lineno)
                    # Use end_lineno if available, otherwise use computed end_line
                    if hasattr(node, "end_lineno") and node.end_lineno:
                        end_line = max(end_line, node.end_lineno)
                    self.body_ranges.append((node.lineno, end_line))
            self.generic_visit(node)
            self.current_class_line = old_class_line

        def visit_FunctionDef(self, node: ast.FunctionDef):
            # Check if it's a method (inside a class) or a function
            if self.current_class_line is not None:
                # It's a method - check if it's unused
                if node.lineno in self.unused_method_lines:
                    # Find the end of the method body
                    if node.body:
                        end_line = node.lineno
                        for stmt in node.body:
                            for child in ast.walk(stmt):
                                if hasattr(child, "end_lineno") and child.end_lineno:
                                    end_line = max(end_line, child.end_lineno)
                                elif hasattr(child, "lineno") and child.lineno:
                                    end_line = max(end_line, child.lineno)
                        # Use end_lineno if available, otherwise use computed end_line
                        if hasattr(node, "end_lineno") and node.end_lineno:
                            end_line = max(end_line, node.end_lineno)
                        self.body_ranges.append((node.lineno, end_line))
            else:
                # It's a function - check if it's unused
                if node.lineno in self.unused_function_lines:
                    # Find the end of the function body
                    if node.body:
                        end_line = node.lineno
                        for stmt in node.body:
                            for child in ast.walk(stmt):
                                if hasattr(child, "end_lineno") and child.end_lineno:
                                    end_line = max(end_line, child.end_lineno)
                                elif hasattr(child, "lineno") and child.lineno:
                                    end_line = max(end_line, child.lineno)
                        # Use end_lineno if available, otherwise use computed end_line
                        if hasattr(node, "end_lineno") and node.end_lineno:
                            end_line = max(end_line, node.end_lineno)
                        self.body_ranges.append((node.lineno, end_line))
            self.generic_visit(node)

        def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef):
            # Check if it's a method (inside a class) or a function
            if self.current_class_line is not None:
                # It's a method - check if it's unused
                if node.lineno in self.unused_method_lines:
                    # Find the end of the method body
                    if node.body:
                        end_line = node.lineno
                        for stmt in node.body:
                            for child in ast.walk(stmt):
                                if hasattr(child, "end_lineno") and child.end_lineno:
                                    end_line = max(end_line, child.end_lineno)
                                elif hasattr(child, "lineno") and child.lineno:
                                    end_line = max(end_line, child.lineno)
                        # Use end_lineno if available, otherwise use computed end_line
                        if hasattr(node, "end_lineno") and node.end_lineno:
                            end_line = max(end_line, node.end_lineno)
                        self.body_ranges.append((node.lineno, end_line))
            else:
                # It's a function - check if it's unused
                if node.lineno in self.unused_function_lines:
                    # Find the end of the function body
                    if node.body:
                        end_line = node.lineno
                        for stmt in node.body:
                            for child in ast.walk(stmt):
                                if hasattr(child, "end_lineno") and child.end_lineno:
                                    end_line = max(end_line, child.end_lineno)
                                elif hasattr(child, "lineno") and child.lineno:
                                    end_line = max(end_line, child.lineno)
                        # Use end_lineno if available, otherwise use computed end_line
                        if hasattr(node, "end_lineno") and node.end_lineno:
                            end_line = max(end_line, node.end_lineno)
                        self.body_ranges.append((node.lineno, end_line))
            self.generic_visit(node)

    visitor = BodyRangeVisitor(unused_class_lines, unused_function_lines, unused_method_lines)
    visitor.visit(tree)
    return visitor.body_ranges


def generate_markdown_report(report: CoverageReport, output_path: Optional[Path] = None) -> Path:
    """
    Generate markdown coverage report with improved dead code detection.

    Args:
        report: CoverageReport to generate report from
        output_path: Optional output path (defaults to COVERAGE_REPORT_TIMESTAMP.md)

    Returns:
        Path to generated report file
    """
    if output_path is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        script_name = Path(report.script_path).stem
        output_path = Path(f"COVERAGE_REPORT_{script_name}_{timestamp}.md")

    # Categorize files by priority
    high_priority_files = []  # Files with both executed and dead code (cleanup opportunities)
    low_priority_files = []  # Files that are completely dead or completely executed

    # Get unused definitions by file for filtering
    unused_by_file: Dict[str, List[UnusedDefinition]] = {}
    for unused in report.unused_definitions:
        if unused.file not in unused_by_file:
            unused_by_file[unused.file] = []
        unused_by_file[unused.file].append(unused)

    for file_path in report.executed_lines.keys():
        dead_count = len(report.dead_lines.get(file_path, []))
        executed_count = len(report.executed_lines.get(file_path, set()))
        executed_lines_set = report.executed_lines.get(file_path, set())
        file_unused = unused_by_file.get(file_path, [])

        # Check if file only has unused definitions (definition lines executed, but no body code)
        # If so, exclude from high-priority (it's already in unused definitions section)
        # Count how many executed lines are definition lines for unused definitions
        if file_unused:
            # Get all definition lines for unused definitions (class/function definitions)
            unused_definition_lines = {unused.definition_line for unused in file_unused}

            # For unused classes, also find method definition lines within those classes
            # Parse the file to find method definitions within unused classes
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    source = f.read()
                tree = ast.parse(source, filename=file_path)

                # Find method definition lines within unused classes AND unused methods
                class MethodDefVisitor(ast.NodeVisitor):
                    def __init__(self, unused_class_lines: Set[int], unused_method_lines: Set[int]):
                        self.unused_class_lines = unused_class_lines
                        self.unused_method_lines = unused_method_lines
                        self.method_def_lines = set()
                        self.current_class_line = None

                    def visit_ClassDef(self, node: ast.ClassDef):
                        if node.lineno in self.unused_class_lines:
                            self.current_class_line = node.lineno
                            self.generic_visit(node)
                            self.current_class_line = None
                        else:
                            self.generic_visit(node)

                    def visit_FunctionDef(self, node: ast.FunctionDef):
                        # Include if method is in unused class OR if method itself is unused
                        if self.current_class_line is not None or node.lineno in self.unused_method_lines:
                            # Method definition within unused class or unused method
                            # Include decorator lines (e.g., @staticmethod)
                            for decorator in node.decorator_list:
                                if hasattr(decorator, "lineno") and decorator.lineno:
                                    self.method_def_lines.add(decorator.lineno)
                            # Include all lines from definition to first body line (covers multi-line signatures)
                            self.method_def_lines.add(node.lineno)
                            # Also include parameter lines if method has a body
                            if node.body:
                                first_body_line = node.body[0].lineno
                                # Add all lines from definition to first body line
                                for line_num in range(node.lineno, first_body_line):
                                    self.method_def_lines.add(line_num)
                        self.generic_visit(node)

                    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef):
                        # Include if method is in unused class OR if method itself is unused
                        if self.current_class_line is not None or node.lineno in self.unused_method_lines:
                            # Method definition within unused class or unused method
                            # Include decorator lines (e.g., @staticmethod)
                            for decorator in node.decorator_list:
                                if hasattr(decorator, "lineno") and decorator.lineno:
                                    self.method_def_lines.add(decorator.lineno)
                            # Include all lines from definition to first body line (covers multi-line signatures)
                            self.method_def_lines.add(node.lineno)
                            # Also include parameter lines if method has a body
                            if node.body:
                                first_body_line = node.body[0].lineno
                                # Add all lines from definition to first body line
                                for line_num in range(node.lineno, first_body_line):
                                    self.method_def_lines.add(line_num)
                        self.generic_visit(node)

                unused_class_lines = {unused.definition_line for unused in file_unused if unused.type == "class"}
                unused_method_lines = {unused.definition_line for unused in file_unused if unused.type == "method"}
                visitor = MethodDefVisitor(unused_class_lines, unused_method_lines)
                visitor.visit(tree)

                # Combine class/function definition lines with method definition lines
                all_unused_def_lines = unused_definition_lines | visitor.method_def_lines
            except Exception:  # noqa: BLE001
                # If parsing fails, fall back to just definition lines
                all_unused_def_lines = unused_definition_lines

            executed_unused_def_lines = all_unused_def_lines & executed_lines_set

            # Exclude import lines and other module-level non-definition code from the count
            # Parse the file to find import lines
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    source = f.read()
                tree = ast.parse(source, filename=file_path)

                # Find import lines (import and from ... import statements)
                import_lines = set()
                for node in ast.walk(tree):
                    if isinstance(node, (ast.Import, ast.ImportFrom)):
                        if hasattr(node, "lineno") and node.lineno:
                            import_lines.add(node.lineno)
                        # Also include continuation lines for multi-line imports
                        if isinstance(node, ast.ImportFrom) and hasattr(node, "names"):
                            for alias in node.names:
                                if hasattr(alias, "lineno") and alias.lineno:
                                    import_lines.add(alias.lineno)
            except Exception:  # noqa: BLE001
                import_lines = set()

            # Count non-import executed lines
            non_import_executed = executed_lines_set - import_lines

            # If ALL non-import executed lines are just definition lines for unused definitions,
            # exclude from high-priority (the class/function/method is imported but never used)
            # No arbitrary threshold - if everything executed is just definitions, it's unused
            if len(non_import_executed) > 0:
                # Check if all non-import executed lines are definition lines for unused definitions
                only_unused_definitions = non_import_executed.issubset(executed_unused_def_lines)
            else:
                only_unused_definitions = False
        else:
            only_unused_definitions = False

        if dead_count > 0 and executed_count > 0 and not only_unused_definitions:
            # Mixed execution - high priority for cleanup
            high_priority_files.append(file_path)
        else:
            low_priority_files.append(file_path)

    # Sort high priority by dead code density (dead lines / total lines)
    high_priority_files.sort(
        key=lambda fp: len(report.dead_lines.get(fp, [])) / max(report.total_lines.get(fp, 1), 1), reverse=True
    )

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("# Coverage Report\n\n")
        f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"**Script:** `{report.script_path}`\n\n")

        f.write("## Summary\n\n")

        # Overall statistics
        total_files = len(report.executed_lines)
        total_executed = sum(len(lines) for lines in report.executed_lines.values())
        total_dead = sum(len(lines) for lines in report.dead_lines.values())

        f.write(f"- **Files analyzed:** {total_files}\n")
        f.write(f"- **Lines executed:** {total_executed}\n")
        f.write(f"- **Dead lines:** {total_dead}\n")
        f.write(f"- **Dead branches:** {len(report.dead_branches)}\n")
        f.write(f"- **Unused definitions (defined but never used):** {len(report.unused_definitions)}\n")
        f.write(f"- **High priority files (mixed execution):** {len(high_priority_files)}\n")
        f.write(f"- **Low priority files (all dead/all executed):** {len(low_priority_files)}\n\n")

        # Unused Definitions Section - Classes/functions defined but never used
        if report.unused_definitions:
            f.write("## ⚠️ Unused Definitions (Defined But Never Used)\n\n")
            f.write(
                "These classes and functions are defined (imported) but never actually used:\n"
                "Only their definition lines were executed, but no method/function bodies.\n\n"
            )

            # Group by file (reuse unused_by_file from above)
            for file_path in sorted(unused_by_file.keys())[:50]:  # Limit to top 50 files
                file_unused = unused_by_file[file_path]
                f.write(f"### `{file_path}`\n\n")

                # Group by type
                classes = [u for u in file_unused if u.type == "class"]
                functions = [u for u in file_unused if u.type == "function"]
                methods = [u for u in file_unused if u.type == "method"]

                if classes:
                    f.write(f"- **Unused classes ({len(classes)}):**\n")
                    for cls in sorted(classes, key=lambda x: x.line):
                        f.write(f"  - `{cls.name}` (line {cls.line})\n")
                    f.write("\n")

                if methods:
                    f.write(f"- **Unused methods ({len(methods)}):**\n")
                    for method in sorted(methods, key=lambda x: x.line):
                        f.write(f"  - `{method.name}` (line {method.line})\n")
                    f.write("\n")

                if functions:
                    f.write(f"- **Unused functions ({len(functions)}):**\n")
                    for func in sorted(functions, key=lambda x: x.line):
                        f.write(f"  - `{func.name}` (line {func.line})\n")
                    f.write("\n")

            remaining = len(report.unused_definitions) - sum(
                len(unused_by_file.get(fp, [])) for fp in sorted(unused_by_file.keys())[:50]
            )
            if remaining > 0:
                f.write(f"... and {remaining} more unused definitions\n\n")

        # High Priority Section - Dead code near executed code
        if high_priority_files:
            f.write("## 🔴 High Priority: Dead Code Near Executed Code\n\n")
            f.write("These files have both executed and dead code - suggesting cleanup opportunities:\n\n")

            for file_path in high_priority_files[:30]:  # Limit to top 30
                coverage_pct = report.coverage_percentage.get(file_path, 0.0)
                dead_count = len(report.dead_lines.get(file_path, []))
                executed_count = len(report.executed_lines.get(file_path, set()))
                executed_lines_set = report.executed_lines.get(file_path, set())

                f.write(f"### `{file_path}`\n\n")
                f.write(f"- **Coverage:** {coverage_pct:.1f}%\n")
                f.write(f"- **Executed lines:** {executed_count}\n")
                f.write(f"- **Dead lines:** {dead_count}\n\n")

                # Find contiguous dead code blocks
                dead_lines_list = sorted(report.dead_lines.get(file_path, []))
                blocks = _find_contiguous_blocks(dead_lines_list)

                # Merge nearby blocks to show full class/method context
                merged_blocks = _merge_nearby_blocks(blocks, max_distance=30)

                # Filter out blocks that belong to unused definitions (already shown in unused definitions section)
                file_unused = unused_by_file.get(file_path, [])
                if file_unused:
                    # Parse file to find body ranges for unused definitions
                    try:
                        with open(file_path, "r", encoding="utf-8") as src:
                            source = src.read()
                        tree = ast.parse(source, filename=file_path)

                        # Find body ranges for unused definitions
                        unused_body_ranges = _find_unused_definition_body_ranges(tree, file_unused)

                        # Filter out blocks that overlap with unused definition bodies
                        filtered_blocks = []
                        for block_start, block_end in merged_blocks:
                            # Check if block overlaps with any unused definition body
                            overlaps_unused = False
                            for unused_start, unused_end in unused_body_ranges:
                                if block_start <= unused_end and block_end >= unused_start:
                                    overlaps_unused = True
                                    break

                            if not overlaps_unused:
                                filtered_blocks.append((block_start, block_end))

                        merged_blocks = filtered_blocks
                    except Exception:  # noqa: BLE001
                        # If parsing fails, show all blocks
                        pass

                if merged_blocks:
                    f.write("**Dead Code Blocks:**\n\n")
                    for block_start, block_end in merged_blocks[:10]:  # Show first 10 blocks
                        if block_start == block_end:
                            f.write(f"- **Line {block_start}** (single line)\n")
                        else:
                            f.write(f"- **Lines {block_start}-{block_end}** ({block_end - block_start + 1} lines)\n")

                        # Show context around the block (show full context for merged blocks)
                        try:
                            with open(file_path, "r", encoding="utf-8") as src:
                                all_lines = src.readlines()

                            # Show context before and after block (more context for merged blocks)
                            context_start = max(0, block_start - 4)
                            context_end = min(len(all_lines), block_end + 3)

                            f.write("  ```python\n")
                            for line_num in range(context_start, context_end):
                                if line_num < len(all_lines):
                                    is_executed = (line_num + 1) in executed_lines_set
                                    is_dead = (line_num + 1) in dead_lines_list
                                    # Use fixed-width format for alignment
                                    # Use emoji for all lines to ensure consistent width
                                    if is_dead:
                                        marker_area = "❌"
                                    elif is_executed:
                                        marker_area = "✅"
                                    else:
                                        # Use white circle emoji - same width as other emojis
                                        marker_area = "⚪"  # White circle - same emoji width as ❌ and ✅
                                    line_content = all_lines[line_num].rstrip()
                                    # Format: 2 spaces + marker_area + space + line_num (4 digits) + " |" + content
                                    f.write(f"  {marker_area} {line_num + 1:4d} | {line_content}\n")
                            f.write("  ```\n\n")
                        except Exception:  # noqa: BLE001
                            pass

                    if len(blocks) > 10:
                        f.write(f"- ... and {len(blocks) - 10} more blocks\n\n")
                    f.write("\n")

        # Dead Branches with Context
        if report.dead_branches:
            f.write("## 🔴 Dead Branches (Near Executed Code)\n\n")
            f.write("These branches were never taken, but their parent code was executed:\n\n")

            # Group branches by file
            branches_by_file: Dict[str, List[BranchInfo]] = {}
            for branch in report.dead_branches:
                if branch.file not in branches_by_file:
                    branches_by_file[branch.file] = []
                branches_by_file[branch.file].append(branch)

            # Only show branches from high-priority files (where code was executed)
            shown_count = 0
            for file_path in high_priority_files:
                if file_path in branches_by_file and shown_count < 30:
                    file_branches = branches_by_file[file_path]
                    executed_lines_set = report.executed_lines.get(file_path, set())

                    for branch in file_branches[:5]:  # Max 5 per file
                        if shown_count >= 30:
                            break

                        f.write(f"### `{branch.file}:{branch.line}`\n\n")
                        f.write(f"- **Type:** {branch.branch_type}\n")
                        f.write(f"- **Condition:** `{branch.condition}`\n")
                        f.write("- **Status:** ❌ Never executed\n")

                        # Show context around the branch
                        try:
                            with open(branch.file, "r", encoding="utf-8") as src:
                                all_lines = src.readlines()

                            context_start = max(0, branch.line - 3)
                            context_end = min(len(all_lines), branch.line + 5)

                            f.write("\n**Context:**\n```python\n")
                            for line_num in range(context_start, context_end):
                                if line_num < len(all_lines):
                                    is_executed = (line_num + 1) in executed_lines_set
                                    is_branch_line = (line_num + 1) == branch.line
                                    # Use fixed-width format for alignment
                                    # Use emoji for all lines to ensure consistent width
                                    if is_branch_line:
                                        marker_area = "❌"
                                    elif is_executed:
                                        marker_area = "✅"
                                    else:
                                        # Use white circle emoji - same width as other emojis
                                        marker_area = "⚪"  # White circle - same emoji width as ❌ and ✅
                                    line_content = all_lines[line_num].rstrip()
                                    # Format: marker_area + space + line_num (4 digits) + " |" + content
                                    f.write(f"{marker_area} {line_num + 1:4d} | {line_content}\n")
                            f.write("```\n\n")
                        except Exception:  # noqa: BLE001
                            pass

                        shown_count += 1

                    if shown_count >= 30:
                        break

            remaining_branches = len(report.dead_branches) - shown_count
            if remaining_branches > 0:
                f.write(f"... and {remaining_branches} more dead branches\n\n")

        # Low Priority Section - All other files
        if low_priority_files:
            f.write("## ⚪ Low Priority: Completely Dead or Executed Files\n\n")
            f.write("These files are either completely unused or fully executed:\n\n")

            for file_path in sorted(low_priority_files):
                coverage_pct = report.coverage_percentage.get(file_path, 0.0)
                dead_count = len(report.dead_lines.get(file_path, []))
                executed_count = len(report.executed_lines.get(file_path, set()))

                f.write(f"### `{file_path}`\n\n")
                f.write(f"- **Coverage:** {coverage_pct:.1f}%\n")
                f.write(f"- **Executed lines:** {executed_count}\n")
                f.write(f"- **Dead lines:** {dead_count}\n\n")

    return output_path
