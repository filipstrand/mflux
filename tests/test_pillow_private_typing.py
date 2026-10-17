import ast
from pathlib import Path

import pytest

# PIL._typing is private to Pillow. Pillow 13.0.0's copy fails to import on Python 3.11 to 3.14
# (`from numbers import _IntegralLike`), which broke mflux on a fresh install. Pillow's own
# modules may still import it, so the guard reads mflux's source instead of blocking the module.
# Only static import statements are checked; a TYPE_CHECKING import is flagged too, on purpose.
_SRC = Path(__file__).resolve().parents[1] / "src" / "mflux"


def _private_typing_import_lines(path: Path) -> list[int]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    lines = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module == "PIL._typing":
            lines.append(node.lineno)
        elif isinstance(node, ast.ImportFrom) and node.module == "PIL" and any(a.name == "_typing" for a in node.names):
            lines.append(node.lineno)
        elif isinstance(node, ast.Import) and any(alias.name == "PIL._typing" for alias in node.names):
            lines.append(node.lineno)
    return lines


@pytest.mark.fast
def test_no_module_imports_pillow_private_typing() -> None:
    assert (_SRC / "utils" / "image_util.py").is_file()
    paths = sorted(_SRC.rglob("*.py"))
    offenders = {
        str(path.relative_to(_SRC.parent)): lines for path in paths if (lines := _private_typing_import_lines(path))
    }
    assert offenders == {}
