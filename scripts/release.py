#!/usr/bin/env python3

import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import Optional

import requests


class ReleaseManager:
    def __init__(self, github_token: str):
        self.github_token = github_token
        self.github_repo = "filipstrand/mflux"
        self.base_url = "https://api.github.com"
        self.headers = {
            "Authorization": f"token {github_token}",
            "Accept": "application/vnd.github.v3+json",
            "User-Agent": "MFLUX-Release-Script",
        }

    @staticmethod
    def extract_version_from_ref(github_ref: str) -> str:
        if not github_ref.startswith("refs/tags/v."):
            raise ValueError(f"Invalid GitHub ref format: {github_ref}")
        return github_ref.replace("refs/tags/v.", "")

    @staticmethod
    def extract_changelog_entry(version: str, changelog_path: Path = Path("CHANGELOG.md")) -> str:
        import re

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

        return entry_content

    def check_release_exists(self, tag_name: str) -> bool:
        url = f"{self.base_url}/repos/{self.github_repo}/releases/tags/{tag_name}"
        response = requests.get(url, headers=self.headers)
        return response.status_code == 200

    def create_github_release(self, tag_name: str, version: str, changelog_entry: str) -> dict:
        url = f"{self.base_url}/repos/{self.github_repo}/releases"

        data = {
            "tag_name": tag_name,
            "name": f"Release {version}",
            "body": changelog_entry,
            "draft": False,
            "prerelease": False,
        }

        response = requests.post(url, json=data, headers=self.headers)

        if response.status_code == 201:
            print(f"✅ Successfully created GitHub release for {tag_name}")
            return response.json()
        else:
            raise Exception(f"Failed to create GitHub release: {response.status_code} - {response.text}")

    @staticmethod
    def run_command(cmd: list, description: str, check: bool = True) -> subprocess.CompletedProcess:
        print(f"🔄 {description}...")
        print(f"   Running: {' '.join(cmd)}")

        result = subprocess.run(cmd, capture_output=True, text=True, check=False)

        if result.returncode != 0 and check:
            print(f"❌ {description} failed!")
            print(f"   stdout: {result.stdout}")
            print(f"   stderr: {result.stderr}")
            raise Exception(f"Command failed: {' '.join(cmd)}")

        if result.stdout:
            print(f"   stdout: {result.stdout.strip()}")

        print(f"✅ {description} completed successfully")
        return result

    def build_package(self):
        # Install build dependencies
        self.run_command(
            [sys.executable, "-m", "pip", "install", "--upgrade", "pip", "build", "twine"],
            "Installing build dependencies",
        )

        # Build the package
        self.run_command([sys.executable, "-m", "build"], "Building package")

    def publish_to_test_pypi(self, test_pypi_token: Optional[str]):
        if not test_pypi_token:
            print("⚠️  Test PyPI token not provided, skipping Test PyPI upload")
            return

        env = os.environ.copy()
        env["TWINE_USERNAME"] = "__token__"
        env["TWINE_PASSWORD"] = test_pypi_token

        self.run_command(
            [sys.executable, "-m", "twine", "upload", "--repository", "testpypi", "dist/*", "--verbose"],
            "Publishing to Test PyPI",
        )

    @staticmethod
    def publish_to_pypi(pypi_token: str):
        env = os.environ.copy()
        env["TWINE_USERNAME"] = "__token__"
        env["TWINE_PASSWORD"] = pypi_token

        # Set environment variables for subprocess
        cmd_env = os.environ.copy()
        cmd_env.update(env)

        print("🔄 Publishing to PyPI...")
        print(f"   Running: {sys.executable} -m twine upload dist/* --verbose")

        result = subprocess.run(
            [sys.executable, "-m", "twine", "upload", "dist/*", "--verbose"],
            capture_output=True,
            text=True,
            env=cmd_env,
            check=False,
        )

        if result.returncode != 0:
            print("❌ Publishing to PyPI failed!")
            print(f"   stdout: {result.stdout}")
            print(f"   stderr: {result.stderr}")
            raise Exception("PyPI upload failed")

        if result.stdout:
            print(f"   stdout: {result.stdout.strip()}")

        print("✅ Publishing to PyPI completed successfully")

    def release(self, version: str, test_pypi_token: Optional[str], pypi_token: str):
        print("🚀 Starting MFLUX release process...")

        # Extract version
        tag_name = f"v.{version}"
        print(f"📦 Releasing version: {version} (tag: {tag_name})")

        # Extract changelog
        print("📝 Extracting changelog entry...")
        changelog_entry = self.extract_changelog_entry(version)
        print(f"   Found changelog entry ({len(changelog_entry)} characters)")

        # Check if release exists
        print("🔍 Checking if release already exists...")
        if self.check_release_exists(tag_name):
            print(f"⚠️  Release {tag_name} already exists, skipping GitHub release creation")
            skip_github_release = True
        else:
            skip_github_release = False

        # Create GitHub release
        if not skip_github_release:
            self.create_github_release(tag_name, version, changelog_entry)

        # Build package
        self.build_package()

        # Publish to Test PyPI
        if not skip_github_release:  # Only publish if this is a new release
            self.publish_to_test_pypi(test_pypi_token)

            # Publish to PyPI
            self.publish_to_pypi(pypi_token)
        else:
            print("⚠️  Skipping PyPI publishing since release already exists")

        print(f"🎉 Release process completed successfully for version {version}!")


def main():
    parser = argparse.ArgumentParser(description="MFLUX Release Management")
    parser.add_argument("--version", help="Release version (e.g., 0.8.0)")
    parser.add_argument("--github-token", help="GitHub token")
    parser.add_argument("--test-pypi-token", help="Test PyPI API token (optional)")
    parser.add_argument("--pypi-token", help="PyPI API token")
    parser.add_argument("--dry-run", action="store_true", help="Print what would be done without executing")
    parser.add_argument("--show-changelog", help="Show changelog entry for a specific version (e.g., 0.8.0) and exit")
    parser.add_argument("--markdown", action="store_true", help="Output changelog in markdown format (use with --show-changelog)")  # fmt: off

    args = parser.parse_args()

    # Handle --show-changelog option
    if args.show_changelog:
        try:
            changelog_entry = ReleaseManager.extract_changelog_entry(args.show_changelog)

            if args.markdown:
                # Output as Markdown
                print(f"# Release {args.show_changelog}\n")
                print(changelog_entry)
                print(f"\n---\n*Changelog entry: {len(changelog_entry)} characters*")
            else:
                # Output with separators (original format)
                print(f"📝 Changelog entry for version {args.show_changelog}:")
                print("=" * 60)
                print(changelog_entry)
                print("=" * 60)
                print(f"✅ Found changelog entry ({len(changelog_entry)} characters)")
        except (ValueError, FileNotFoundError) as e:
            print(f"❌ Failed to extract changelog: {e}")
            sys.exit(1)
        return

    # Get values from args or environment variables
    version = args.version or os.getenv("VERSION")
    github_token = args.github_token or os.getenv("GITHUB_TOKEN")
    test_pypi_token = args.test_pypi_token or os.getenv("TEST_PYPI_API_TOKEN")
    pypi_token = args.pypi_token or os.getenv("PYPI_API_TOKEN")

    # Validate required parameters
    if not version:
        print("❌ Version is required (--version or VERSION env var)")
        sys.exit(1)

    if not github_token:
        print("❌ GitHub token is required (--github-token or GITHUB_TOKEN env var)")
        sys.exit(1)

    if not pypi_token:
        print("❌ PyPI token is required (--pypi-token or PYPI_API_TOKEN env var)")
        sys.exit(1)

    if args.dry_run:
        print("🔍 DRY RUN MODE - would execute release process with:")
        print(f"   Version: {version}")
        print(f"   Has Test PyPI token: {bool(test_pypi_token)}")
        print(f"   Has PyPI token: {bool(pypi_token)}")
        return

    try:
        release_manager = ReleaseManager(github_token)
        release_manager.release(version, test_pypi_token, pypi_token)
    except (ValueError, FileNotFoundError, requests.RequestException) as e:
        print(f"❌ Release failed: {e}")
        sys.exit(1)
    except Exception as e:  # noqa: BLE001
        print(f"❌ Release failed with unexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
