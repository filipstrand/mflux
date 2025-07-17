from __future__ import annotations

import importlib.metadata
from pathlib import Path

import toml


class VersionUtil:
    @staticmethod
    def get_mflux_version() -> str:
        return VersionUtil._scan_pyproject() or VersionUtil._get_installed_version() or "unknown"

    @staticmethod
    def _scan_pyproject() -> str | None:
        current_dir = Path(__file__).resolve().parent
        for parent in current_dir.parents:
            pyproject_path = parent / "pyproject.toml"
            if pyproject_path.exists():
                try:
                    data = toml.load(pyproject_path)
                    return data.get("project", {}).get("version")
                except Exception:  # noqa: BLE001
                    return None
        return None

    @staticmethod
    def _get_installed_version() -> str | None:
        try:
            return importlib.metadata.version("mflux")
        except importlib.metadata.PackageNotFoundError:
            return None
