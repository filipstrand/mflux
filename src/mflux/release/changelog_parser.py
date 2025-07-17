import re
from pathlib import Path
from typing import List, Tuple


class ChangelogParser:
    @staticmethod
    def extract_release_notes_from_changelog(version: str, changelog_path: Path = Path("CHANGELOG.md")) -> str:
        print("📝 Extracting changelog entry...")
        if not changelog_path.exists():
            raise FileNotFoundError(f"Changelog file not found: {changelog_path}")
        content = changelog_path.read_text(encoding="utf-8")

        # Pattern to match version headers: ## [VERSION]
        version_pattern = rf"^## \[{re.escape(version)}\].*$"
        next_version_pattern = r"^## \[.*\].*$"

        lines = content.splitlines()

        # Find the start of the target version section
        start_idx = None
        for i, line in enumerate(lines):
            if re.match(version_pattern, line):
                start_idx = i + 1  # Start after the version header
                break

        if start_idx is None:
            raise ValueError(f"Version {version} not found in changelog")

        # Find the end of the section (next version header)
        end_idx = len(lines)
        for i in range(start_idx, len(lines)):
            if re.match(next_version_pattern, lines[i]):
                end_idx = i
                break

        # Extract the content between version headers
        entry_lines = lines[start_idx:end_idx]

        # Clean up: remove empty lines at start/end and strip trailing whitespace
        while entry_lines and not entry_lines[0].strip():
            entry_lines.pop(0)
        while entry_lines and not entry_lines[-1].strip():
            entry_lines.pop()

        entry_content = "\n".join(line.rstrip() for line in entry_lines)

        if not entry_content.strip():
            raise ValueError(f"No content found for version {version}")

        print(f"   Found changelog entry ({len(entry_content)} characters)")
        return entry_content

    @staticmethod
    def validate_changelog_format(changelog_path: Path = Path("CHANGELOG.md")) -> List[str]:
        if not changelog_path.exists():
            raise FileNotFoundError(f"Changelog file not found: {changelog_path}")

        content = changelog_path.read_text(encoding="utf-8")
        lines = content.splitlines()
        issues = []

        # Check if file starts with proper header
        if not lines or not lines[0].strip().startswith("# Changelog"):
            issues.append("Changelog should start with '# Changelog' header")

        # Check for proper version format
        version_pattern = r"^## \[([^\]]+)\](?:\s*-\s*(.+))?.*$"
        found_versions = []

        for i, line in enumerate(lines):
            if re.match(r"^## \[", line):
                match = re.match(version_pattern, line)
                if not match:
                    issues.append(f"Line {i + 1}: Invalid version header format: {line}")
                else:
                    version = match.group(1)
                    found_versions.append(version)

                    # Check version format (except for "Unreleased")
                    if version.lower() != "unreleased":
                        if not re.match(r"^\d+\.\d+\.\d+", version):
                            issues.append(f"Line {i + 1}: Version should follow semantic versioning: {version}")

        if not found_versions:
            issues.append("No version sections found in changelog")

        return issues

    @staticmethod
    def get_latest_version(changelog_path: Path = Path("CHANGELOG.md")) -> str:
        versions = ChangelogParser._list_all_versions(changelog_path)
        if not versions:
            raise ValueError("No versions found in changelog")
        return versions[0][0]  # First version in list (should be latest)

    @staticmethod
    def _list_all_versions(changelog_path: Path = Path("CHANGELOG.md")) -> List[Tuple[str, str]]:
        if not changelog_path.exists():
            raise FileNotFoundError(f"Changelog file not found: {changelog_path}")

        content = changelog_path.read_text(encoding="utf-8")
        lines = content.splitlines()

        versions = []
        version_pattern = r"^## \[([^\]]+)\](?:\s*-\s*(.+))?.*$"

        for line in lines:
            match = re.match(version_pattern, line)
            if match:
                version = match.group(1)
                date = match.group(2) if match.group(2) else "No date"
                if version.lower() != "unreleased":  # Skip "Unreleased" section
                    versions.append((version, date))

        return versions
