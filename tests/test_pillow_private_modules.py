import ast
from pathlib import Path

import pytest

pytestmark = pytest.mark.fast

SRC = Path(__file__).parents[1] / "src" / "mflux"


def _runtime_private_pil_imports(tree: ast.AST) -> list[tuple[int, str]]:
    found = []
    for node in ast.walk(tree):
        if isinstance(node, ast.If) and isinstance(node.test, ast.Name) and node.test.id == "TYPE_CHECKING":
            node.body = []  # imports only seen by type checkers are fine
    for node in ast.walk(tree):
        names = []
        if isinstance(node, ast.Import):
            names = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom) and node.module == "PIL":
            names = [f"PIL.{alias.name}" for alias in node.names]
        elif isinstance(node, ast.ImportFrom) and node.module:
            names = [node.module]
        found += [(node.lineno, name) for name in names if name.startswith("PIL._")]
    return found


def test_no_runtime_imports_of_private_pillow_modules():
    # Pillow 13's private PIL._typing raises ImportError below Python 3.15, which crashed every command at
    # import. Pillow's private modules are not an API; define what is needed in mflux instead.
    offenders = [
        f"{path.relative_to(SRC.parent)}:{line} imports {name}"
        for path in sorted(SRC.rglob("*.py"))
        for line, name in _runtime_private_pil_imports(ast.parse(path.read_text(), str(path)))
    ]
    assert not offenders, offenders
