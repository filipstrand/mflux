import shutil
import sys
import time
from pathlib import Path

import requests
from twine.commands import upload
from twine.exceptions import TwineException
from twine.settings import Settings

from .git_operations import GitOperations


class PyPIPublisher:
    @staticmethod
    def build_and_verify_package() -> None:
        print("📦 Building and verifying package...")

        # Clean dist directory
        dist_dir = Path("dist")
        if dist_dir.exists():
            shutil.rmtree(dist_dir)
            print("🧹 Cleaned dist/ directory")

        # Build the package
        GitOperations.run_command([sys.executable, "-m", "uv", "--version"], "Verify 'uv build' version")
        GitOperations.run_command([sys.executable, "-m", "uv", "build"], "Building package with 'uv build'")

        # Verify the package
        print("🔍 Verifying package...")
        GitOperations.run_command(
            [sys.executable, "-m", "twine", "check", "dist/*"], "Verifying distribution with Twine"
        )

    @staticmethod
    def version_exists_on_pypi(package_name: str, version: str) -> bool:
        print("🔍 Checking if version already exists on PyPI...")
        repo_url = "https://pypi.org/pypi"
        url = f"{repo_url}/{package_name}/json"

        for attempt in range(3):
            try:
                print(f"   Checking PyPI API (attempt {attempt + 1}/3): {url}")
                response = requests.get(url, timeout=30)

                if response.status_code == 200:
                    data = response.json()
                    releases = data.get("releases", {})
                    version_exists = version in releases
                    print(
                        f"   PyPI API response: {len(releases)} total versions, version {version} exists: {version_exists}"
                    )

                    if version_exists:
                        print(f"⚠️  Version {version} already exists on PyPI")

                    return version_exists
                elif response.status_code == 404:
                    print("   Package not found on PyPI (404) - this is normal for new packages")
                    return False
                else:
                    print(f"   PyPI API returned status {response.status_code}: {response.text[:200]}")
                    if attempt < 2:
                        time.sleep(2**attempt)
                        continue
                    else:
                        raise ValueError(f"PyPI API returned unexpected status {response.status_code}")

            except requests.RequestException as e:
                print(f"   Network error checking PyPI (attempt {attempt + 1}/3): {e}")
                if attempt < 2:
                    time.sleep(2**attempt)
                    continue
                else:
                    raise requests.RequestException(f"Failed to check PyPI after 3 attempts: {e}") from e

        raise RuntimeError("Unexpected error in version_exists_on_pypi")

    @staticmethod
    def publish_to_pypi(pypi_token: str, package_name: str, version: str) -> None:
        PyPIPublisher._upload_to_pypi(
            token=pypi_token,
            repository="pypi",
            display_name="PyPI",
            package_name=package_name,
            version=version,
            optional=False,
        )

    @staticmethod
    def _upload_to_pypi(
        token: str,
        repository: str,
        display_name: str,
        package_name: str,
        version: str,
        optional: bool = False,
    ) -> None:
        print(f"📦 Publishing to {display_name}...")
        try:
            print(f"🔄 Using programmatic twine upload for {display_name}...")

            settings = Settings(
                username="__token__",
                password=token,
                repository=repository,
                verbose=True,
                skip_existing=True,
                disable_progress_bar=False,
                comment="Automated release via mflux-release-script",
            )

            # Only select valid distribution files (.whl and .tar.gz)
            dist_files = []
            dist_dir = Path("dist")
            dist_files.extend(dist_dir.glob("*.whl"))
            dist_files.extend(dist_dir.glob("*.tar.gz"))

            if not dist_files:
                raise ValueError("No distribution files found in dist/")

            print(f"📦 Uploading {len(dist_files)} files to {display_name}...")
            for file_path in dist_files:
                print(f"   • {file_path.name}")

            # Retry transient failures (e.g. 5xx replies / connection resets)
            for attempt in range(3):
                try:
                    upload.upload(settings, [str(f) for f in dist_files])
                    print(f"✅ Programmatic {display_name} upload completed successfully (attempt {attempt + 1})")
                    break  # success → leave retry loop
                except TwineException as te:
                    transient = any(x in str(te).lower() for x in ["500", "502", "503", "504", "timeout"])
                    if transient and attempt < 2:
                        wait = 2**attempt
                        print(
                            f"⚠️  Transient {display_name} upload error (attempt {attempt + 1}/3): {te}. Retrying in {wait}s"
                        )
                        time.sleep(wait)
                        continue
                    raise  # re-raise to outer handler for consistent processing
        except TwineException as e:
            error_msg = str(e).lower()
            if "already exists" in error_msg:
                print(f"⚠️  Version {version} already exists on {display_name}")
                return
            else:
                # Handle all other TwineException errors consistently
                print(f"⚠️  {display_name} upload failed: {e}")
                if "authentication" in error_msg or "403" in error_msg:
                    print(f"   This appears to be an authentication issue - check your {display_name} token.")
                elif "400" in error_msg or "bad request" in error_msg:
                    print("   This might be a partial upload where some files (like wheels) succeeded.")
                    print("   If only wheel uploaded, this is a known issue with source distribution validation.")
                else:
                    print("   This might be a partial upload where some files succeeded.")
                print(f"   Check {display_name} manually to verify which files were uploaded.")
                print(f"   {display_name} failures are non-critical, continuing with release process")
                return
        except (OSError, ValueError, RuntimeError) as e:
            print(f"⚠️  Unexpected error during {display_name} upload: {e}")
            print("   This might be a partial upload where some files succeeded.")
            print(f"   Check {display_name} manually to verify which files were uploaded.")
            print(f"   {display_name} failures are non-critical, continuing with release process")
            return
