#!/usr/bin/env python3

import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import Optional

import requests
from generate_release_notes import ChangelogParser


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
        # Package name on (Test)PyPI – derive from repo slug for now
        self.package_name = self.github_repo.split("/")[1]

    @staticmethod
    def get_version_from_pyproject(pyproject_path: Path = Path("pyproject.toml")) -> str:
        """Read version from pyproject.toml"""
        if not pyproject_path.exists():
            raise FileNotFoundError(f"pyproject.toml not found at {pyproject_path}")

        import toml

        try:
            pyproject_data = toml.load(pyproject_path)
            version = pyproject_data.get("project", {}).get("version")
            if not version:
                raise ValueError("Version not found in pyproject.toml under [project] section")
            return version
        except (toml.TomlDecodeError, KeyError, TypeError) as e:
            raise ValueError(f"Failed to read version from pyproject.toml: {e}")

    @staticmethod
    def validate_release_ready(version: str) -> None:
        """Validate that everything is ready for release"""
        # Check if version looks valid
        import re

        if not re.match(r"^\d+\.\d+\.\d+", version):
            raise ValueError(f"Version format appears invalid: {version}")

            # Check if changelog has entry for this version and validate version consistency
        try:
            changelog_parser = ChangelogParser()
            changelog_parser.extract_changelog_entry(version)  # Just validate entry exists
            print(f"✅ Changelog entry found for version {version}")

            # Validate that pyproject.toml version matches the latest version in changelog
            latest_changelog_version = changelog_parser.get_latest_version()
            if version != latest_changelog_version:
                raise ValueError(
                    f"Version mismatch: pyproject.toml has version '{version}' but latest changelog version is '{latest_changelog_version}'. "
                    f"Please ensure pyproject.toml version matches the latest changelog entry."
                )
            print(f"✅ Version consistency validated: pyproject.toml ({version}) matches latest changelog entry")

        except ValueError as e:
            raise ValueError(f"Changelog validation failed: {e}")

        # Check that we are releasing from the main branch.
        # In local runs we can use `git branch --show-current`, but inside GitHub Actions we are in a
        # detached-HEAD state.  In that case `GITHUB_REF_NAME` (or `GITHUB_HEAD_REF` for PRs) exposes
        # the branch name.
        current_branch = os.getenv("GITHUB_REF_NAME") or os.getenv("GITHUB_HEAD_REF")

        if not current_branch:
            # Fallback to local git query
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

        print(f"✅ Release validation passed for version {version}")

    @staticmethod
    def extract_version_from_ref(github_ref: str) -> str:
        if not github_ref.startswith("refs/tags/v."):
            raise ValueError(f"Invalid GitHub ref format: {github_ref}")
        return github_ref.replace("refs/tags/v.", "")

    def check_release_exists(self, tag_name: str) -> bool:
        url = f"{self.base_url}/repos/{self.github_repo}/releases/tags/{tag_name}"
        response = requests.get(url, headers=self.headers)
        return response.status_code == 200

    def check_git_tag_exists(self, tag_name: str) -> bool:
        """Check if a git tag already exists locally or remotely"""
        try:
            # Check if tag exists locally
            result = subprocess.run(["git", "tag", "-l", tag_name], capture_output=True, text=True, check=True)
            if result.stdout.strip() == tag_name:
                print(f"✅ Git tag {tag_name} exists locally")
                return True

            # Check if tag exists on remote
            result = subprocess.run(
                ["git", "ls-remote", "--tags", "origin", f"refs/tags/{tag_name}"],
                capture_output=True,
                text=True,
                check=True,
            )
            if result.stdout.strip():
                print(f"✅ Git tag {tag_name} exists on remote")
                return True

            return False
        except subprocess.CalledProcessError as e:
            print(f"⚠️  Warning: Could not check git tag existence: {e}")
            return False

    def create_git_tag(self, tag_name: str, version: str) -> None:
        """Create and push git tag for the release"""
        # Create annotated tag
        self.run_command(["git", "tag", "-a", tag_name, "-m", f"Release {version}"], f"Creating git tag {tag_name}")

        # Push tag to remote
        self.run_command(["git", "push", "origin", tag_name], f"Pushing git tag {tag_name} to remote")

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
    def run_command(cmd: list, description: str, check: bool = True, env: dict = None) -> subprocess.CompletedProcess:
        print(f"🔄 {description}...")
        print(f"   Running: {' '.join(cmd)}")

        result = subprocess.run(cmd, capture_output=True, text=True, check=False, env=env)

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

        # Clean dist directory to ensure fresh build
        import shutil

        dist_dir = Path("dist")
        if dist_dir.exists():
            shutil.rmtree(dist_dir)
            print("🧹 Cleaned dist/ directory")

        # Build the package
        self.run_command([sys.executable, "-m", "build"], "Building package")

    def verify_package(self):
        """Run *twine check* to verify that README / long_description renders on PyPI."""
        self.run_command([sys.executable, "-m", "twine", "check", "dist/*"], "Verifying distribution with Twine")

    def publish_to_test_pypi(self, test_pypi_token: Optional[str], version: str):
        if not test_pypi_token:
            print("⚠️  Test PyPI token not provided, skipping Test PyPI upload")
            return

        if self.version_exists_on_pypi(version, test_pypi=True):
            print(f"⚠️  Version {version} already exists on Test PyPI, skipping upload")
            return

        env = os.environ.copy()
        env["TWINE_USERNAME"] = "__token__"
        env["TWINE_PASSWORD"] = test_pypi_token

        self.run_command(
            [sys.executable, "-m", "twine", "upload", "--repository", "testpypi", "dist/*", "--verbose"],
            "Publishing to Test PyPI",
            env=env,
        )

    def publish_to_pypi(self, pypi_token: str, version: str):
        if self.version_exists_on_pypi(version, test_pypi=False):
            print(f"⚠️  Version {version} already exists on PyPI, skipping upload")
            return

        env = os.environ.copy()
        env["TWINE_USERNAME"] = "__token__"
        env["TWINE_PASSWORD"] = pypi_token

        self.run_command(
            [sys.executable, "-m", "twine", "upload", "dist/*", "--verbose"],
            "Publishing to PyPI",
            env=env,
        )

    def version_exists_on_pypi(self, version: str, test_pypi: bool = False) -> bool:
        """Return True if *version* of this package is already published on (Test)PyPI."""
        repo_url = "https://test.pypi.org/pypi" if test_pypi else "https://pypi.org/pypi"
        url = f"{repo_url}/{self.package_name}/json"
        try:
            response = requests.get(url, timeout=10)
            if response.status_code != 200:
                return False
            data = response.json()
            return version in data.get("releases", {})
        except requests.RequestException:
            return False

    def release(self, test_pypi_token: Optional[str], pypi_token: str):
        print("🚀 Starting MFLUX release process...")

        # Read version from pyproject.toml
        version = self.get_version_from_pyproject()
        tag_name = f"v.{version}"
        print(f"📦 Releasing version: {version} (tag: {tag_name}) [from pyproject.toml]")

        # Validate everything is ready for release
        print("🔍 Validating release readiness...")
        self.validate_release_ready(version)

        # Check for uncommitted changes
        print("🔍 Checking for uncommitted changes...")
        try:
            result = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True, check=True)
            if result.stdout.strip():
                print("⚠️  Uncommitted changes detected:")
                print(result.stdout)
                raise ValueError("Cannot release with uncommitted changes. Please commit or stash your changes first.")
            print("✅ No uncommitted changes found")
        except subprocess.CalledProcessError:
            print("⚠️  Warning: Could not check git status")

        # Check if release exists
        print("🔍 Checking if release already exists...")
        if self.check_release_exists(tag_name):
            print(f"⚠️  Release {tag_name} already exists, skipping GitHub release creation")
            skip_github_release = True
        else:
            skip_github_release = False

        # Check if git tag exists
        print("🔍 Checking if git tag already exists...")
        git_tag_exists = self.check_git_tag_exists(tag_name)

        # Determine if this is a completely new release or if we're re-running
        if skip_github_release and git_tag_exists:
            print(f"✅ Release {tag_name} already exists completely (both git tag and GitHub release)")
            print("🔄 This appears to be a re-run of an existing release - nothing to do!")
            print("   If you want to republish to PyPI, delete the GitHub release first.")
            return
        elif skip_github_release or git_tag_exists:
            print("⚠️  Partial release state detected:")
            print(f"   Git tag exists: {git_tag_exists}")
            print(f"   GitHub release exists: {skip_github_release}")
            print("   Will complete the missing parts...")

        # Extract changelog
        print("📝 Extracting changelog entry...")
        changelog_parser = ChangelogParser()
        changelog_entry = changelog_parser.extract_changelog_entry(version)
        print(f"   Found changelog entry ({len(changelog_entry)} characters)")

        # Create git tag if it doesn't exist
        if not git_tag_exists:
            print("🏷️  Creating git tag...")
            self.create_git_tag(tag_name, version)
        else:
            print(f"✅ Git tag {tag_name} already exists, skipping creation")

        # Create GitHub release if it doesn't exist
        if not skip_github_release:
            print("🐙 Creating GitHub release...")
            self.create_github_release(tag_name, version, changelog_entry)
        else:
            print(f"✅ GitHub release {tag_name} already exists, skipping creation")

        # Build & verify package
        self.build_package()
        self.verify_package()

        # Only publish to PyPI if this is a new release (both git tag and GitHub release were created)
        if not skip_github_release and not git_tag_exists:
            print("📦 Publishing to PyPI (new release)...")
            # Publish to Test PyPI
            self.publish_to_test_pypi(test_pypi_token, version)
            # Publish to PyPI
            self.publish_to_pypi(pypi_token, version)
        else:
            print("⚠️  Skipping PyPI publishing since this appears to be a re-run")
            print("   (Either git tag or GitHub release already existed)")
            print("   If you need to republish to PyPI, delete both the git tag and GitHub release first.")

        print(f"🎉 Release process completed successfully for version {version}!")


def main():
    parser = argparse.ArgumentParser(description="MFLUX Release Management")
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
            changelog_parser = ChangelogParser()
            changelog_entry = changelog_parser.extract_changelog_entry(args.show_changelog)

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
    github_token = args.github_token or os.getenv("GITHUB_TOKEN")
    test_pypi_token = args.test_pypi_token or os.getenv("TEST_PYPI_API_TOKEN")
    pypi_token = args.pypi_token or os.getenv("PYPI_API_TOKEN")

    # Validate required parameters
    if not github_token:
        print("❌ GitHub token is required (--github-token or GITHUB_TOKEN env var)")
        sys.exit(1)

    if not pypi_token:
        print("❌ PyPI token is required (--pypi-token or PYPI_API_TOKEN env var)")
        sys.exit(1)

    if args.dry_run:
        try:
            version = ReleaseManager.get_version_from_pyproject()
            print("🔍 DRY RUN MODE - would execute release process with:")
            print(f"   Version: {version} [from pyproject.toml]")
            print(f"   Has Test PyPI token: {bool(test_pypi_token)}")
            print(f"   Has PyPI token: {bool(pypi_token)}")
            return
        except (ValueError, FileNotFoundError) as e:
            print(f"❌ Could not read version for dry run: {e}")
            sys.exit(1)

    try:
        release_manager = ReleaseManager(github_token)
        release_manager.release(test_pypi_token, pypi_token)
    except (ValueError, FileNotFoundError, requests.RequestException) as e:
        print(f"❌ Release failed: {e}")
        sys.exit(1)
    except Exception as e:  # noqa: BLE001
        print(f"❌ Release failed with unexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
