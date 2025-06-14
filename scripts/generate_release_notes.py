#!/usr/bin/env python3

import argparse
import re
import sys
from pathlib import Path
from typing import List, Tuple


class ChangelogParser:
    def __init__(self, changelog_path: Path = Path("CHANGELOG.md")):
        self.changelog_path = changelog_path
        if not self.changelog_path.exists():
            raise FileNotFoundError(f"Changelog file not found: {self.changelog_path}")

    def extract_changelog_entry(self, version: str) -> str:
        """Extract changelog entry for a specific version"""
        content = self.changelog_path.read_text(encoding="utf-8")

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

        return entry_content

    def list_all_versions(self) -> List[Tuple[str, str]]:
        """List all versions in the changelog with their dates"""
        content = self.changelog_path.read_text(encoding="utf-8")
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

    def validate_changelog_format(self) -> List[str]:
        """Validate changelog format and return list of issues"""
        content = self.changelog_path.read_text(encoding="utf-8")
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

    def get_latest_version(self) -> str:
        """Get the latest version from changelog (excluding Unreleased)"""
        versions = self.list_all_versions()
        if not versions:
            raise ValueError("No versions found in changelog")
        return versions[0][0]  # First version in list (should be latest)


def main():
    parser = argparse.ArgumentParser(description="MFLUX Changelog Parser and Release Notes Generator")
    parser.add_argument("--changelog", default="CHANGELOG.md", help="Path to changelog file")

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Show changelog entry command
    show_parser = subparsers.add_parser("show", help="Show changelog entry for a version")
    show_parser.add_argument("version", help="Version to show (e.g., 0.8.0)")
    show_parser.add_argument("--markdown", action="store_true", help="Output in markdown format")

    # List versions command
    subparsers.add_parser("list", help="List all versions in changelog")

    # Validate command
    subparsers.add_parser("validate", help="Validate changelog format")

    # Latest version command
    subparsers.add_parser("latest", help="Show latest version")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    try:
        changelog_parser = ChangelogParser(Path(args.changelog))

        if args.command == "show":
            entry = changelog_parser.extract_changelog_entry(args.version)

            if args.markdown:
                print(f"# Release {args.version}\n")
                print(entry)
                print(f"\n---\n*Changelog entry: {len(entry)} characters*")
            else:
                print(f"📝 Changelog entry for version {args.version}:")
                print("=" * 60)
                print(entry)
                print("=" * 60)
                print(f"✅ Found changelog entry ({len(entry)} characters)")

        elif args.command == "list":
            versions = changelog_parser.list_all_versions()
            print("📋 Available versions in changelog:")
            print("=" * 50)
            for version, date in versions:
                print(f"  {version:<15} - {date}")
            print(f"\n✅ Found {len(versions)} versions")

        elif args.command == "validate":
            issues = changelog_parser.validate_changelog_format()
            if issues:
                print("❌ Changelog validation failed:")
                for issue in issues:
                    print(f"  • {issue}")
                sys.exit(1)
            else:
                print("✅ Changelog format is valid")

        elif args.command == "latest":
            latest = changelog_parser.get_latest_version()
            print(f"📦 Latest version: {latest}")

    except (ValueError, FileNotFoundError) as e:
        print(f"❌ Error: {e}")
        sys.exit(1)
    except (OSError, ImportError, KeyboardInterrupt) as e:
        print(f"❌ Unexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
