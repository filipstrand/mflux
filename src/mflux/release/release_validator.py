import os
import re
import subprocess

from .changelog_parser import ChangelogParser


class ReleaseValidator:
    @staticmethod
    def validate_release_ready(version: str) -> None:
        print("🔍 Validating release readiness...")
        ReleaseValidator._validate_version_format(version)
        ReleaseValidator._validate_changelog_format()
        ReleaseValidator._validate_changelog_entry(version)
        ReleaseValidator._validate_branch()
        ReleaseValidator._validate_uncommitted_changes()
        print(f"✅ Release validation passed for version {version}")

    @staticmethod
    def _validate_version_format(version: str) -> None:
        if not re.match(r"^\d+\.\d+\.\d+", version):
            raise ValueError(f"Version format appears invalid: {version}")

    @staticmethod
    def _validate_changelog_entry(version: str) -> None:
        try:
            ChangelogParser.extract_release_notes_from_changelog(version)
            print(f"✅ Changelog entry found for version {version}")

            # Validate version consistency
            latest_changelog_version = ChangelogParser.get_latest_version()
            if version != latest_changelog_version:
                raise ValueError(
                    f"Version mismatch: pyproject.toml has version '{version}' but "
                    f"latest changelog version is '{latest_changelog_version}'. "
                    f"Please ensure pyproject.toml version matches the latest changelog entry."
                )
            print(f"✅ Version consistency validated: pyproject.toml ({version}) matches latest changelog entry")
        except ValueError as e:
            raise ValueError(f"Changelog validation failed: {e}")

    @staticmethod
    def _validate_branch() -> None:
        current_branch = os.getenv("GITHUB_REF_NAME") or os.getenv("GITHUB_HEAD_REF")

        if not current_branch:
            try:
                result = subprocess.run(["git", "branch", "--show-current"], capture_output=True, text=True, check=True)
                current_branch = result.stdout.strip()
            except subprocess.CalledProcessError:
                current_branch = ""

        if current_branch != "main":
            raise ValueError(
                f"Release must be from 'main' branch, currently on '{current_branch or 'UNKNOWN'}'. "
                "Please switch to main branch first or ensure the workflow checks out 'main'."
            )

        print(f"✅ On main branch ({current_branch})")

    @staticmethod
    def _validate_changelog_format() -> None:
        try:
            issues = ChangelogParser.validate_changelog_format()
            if issues:
                print("❌ Changelog format validation failed:")
                for issue in issues:
                    print(f"   • {issue}")
                raise ValueError(f"Changelog format validation failed with {len(issues)} issues")
            print("✅ Changelog format validation passed")
        except ValueError as e:
            if "Changelog format validation failed" in str(e):
                raise  # Re-raise our own validation error
            else:
                raise ValueError(f"Changelog format validation failed: {e}")

    @staticmethod
    def _validate_uncommitted_changes() -> None:
        try:
            result = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True, check=True)
            if result.stdout.strip():
                print("⚠️  Uncommitted changes detected:")
                print(result.stdout)
                raise ValueError("Cannot release with uncommitted changes. Please commit or stash your changes first.")
            print("✅ No uncommitted changes found")
        except subprocess.CalledProcessError:
            print("⚠️  Warning: Could not check git status")
