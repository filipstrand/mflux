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
class CoverageReport:
    """Coverage analysis report."""

    script_path: str
    executed_lines: Dict[str, Set[int]]  # file -> set of executed line numbers
    total_lines: Dict[str, int]  # file -> total executable lines
    coverage_percentage: Dict[str, float]  # file -> coverage percentage
    dead_lines: Dict[str, List[int]]  # file -> list of unexecuted lines
    branches: List[BranchInfo]  # All branches found
    dead_branches: List[BranchInfo]  # Branches never taken


class CoverageAnalyzer:
    """Analyzes coverage data to find dead code."""

    def __init__(self, coverage_data: Dict[str, Set[int]]):
        """
        Initialize analyzer with coverage data.

        Args:
            coverage_data: Dictionary mapping file paths to sets of executed line numbers
        """
        self.coverage_data = coverage_data

    def analyze_file(self, file_path: str) -> Tuple[List[int], List[BranchInfo]]:
        """
        Analyze a single file for dead code.

        Args:
            file_path: Path to Python file

        Returns:
            Tuple of (dead_lines, branches)
        """
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                source = f.read()
        except Exception:  # noqa: BLE001
            return [], []

        # Parse AST to find executable lines and branches
        try:
            tree = ast.parse(source, filename=file_path)
        except SyntaxError:
            return [], []

        # Find all executable lines
        executable_lines = self._find_executable_lines(tree, source)

        # Find executed lines for this file
        executed_lines = self.coverage_data.get(file_path, set())

        # Find dead lines
        dead_lines = [line for line in executable_lines if line not in executed_lines]

        # Find branches
        branches = self._find_branches(tree, file_path, executed_lines)

        return dead_lines, branches

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

        for file_path in files_to_analyze:
            file_dead_lines, file_branches = self.analyze_file(file_path)

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
        )


def generate_markdown_report(report: CoverageReport, output_path: Optional[Path] = None) -> Path:
    """
    Generate markdown coverage report.

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
        f.write(f"- **Dead branches:** {len(report.dead_branches)}\n\n")

        # Per-file coverage
        f.write("## File Coverage\n\n")
        for file_path in sorted(report.executed_lines.keys()):
            coverage_pct = report.coverage_percentage.get(file_path, 0.0)
            dead_count = len(report.dead_lines.get(file_path, []))
            executed_count = len(report.executed_lines.get(file_path, set()))

            f.write(f"### `{file_path}`\n\n")
            f.write(f"- **Coverage:** {coverage_pct:.1f}%\n")
            f.write(f"- **Executed lines:** {executed_count}\n")
            f.write(f"- **Dead lines:** {dead_count}\n\n")

            # Show dead lines
            if dead_count > 0:
                dead_lines_list = sorted(report.dead_lines[file_path])
                f.write("**Dead lines:**\n")
                # Show first 20 dead lines
                for line_num in dead_lines_list[:20]:
                    try:
                        with open(file_path, "r", encoding="utf-8") as src:
                            lines = src.readlines()
                            if line_num <= len(lines):
                                line_content = lines[line_num - 1].strip()
                                f.write(f"- Line {line_num}: `{line_content[:80]}`\n")
                    except Exception:  # noqa: BLE001, PERF203
                        f.write(f"- Line {line_num}\n")

                if len(dead_lines_list) > 20:
                    f.write(f"- ... and {len(dead_lines_list) - 20} more\n")
                f.write("\n")

        # Dead branches
        if report.dead_branches:
            f.write("## Dead Branches\n\n")
            for branch in report.dead_branches[:50]:  # Limit to first 50
                f.write(f"### `{branch.file}:{branch.line}`\n\n")
                f.write(f"- **Type:** {branch.branch_type}\n")
                f.write(f"- **Condition:** `{branch.condition}`\n")
                f.write("- **Status:** ❌ Never executed\n\n")

            if len(report.dead_branches) > 50:
                f.write(f"... and {len(report.dead_branches) - 50} more dead branches\n\n")

    return output_path
