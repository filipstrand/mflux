"""
CLI for profiling and pruning unused files from transformers/diffusers repos.

Usage:
    mflux-debug-prune setup
    mflux-debug-prune prune <script>
"""

import argparse
import subprocess
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Optional

from mflux_debugger.pruner import analyze_profile, generate_markdown_report, prune_files


def detect_repo_paths() -> tuple[Optional[Path], Optional[Path]]:
    """
    Detect transformers and diffusers repo paths from editable installs.

    Returns:
        Tuple of (transformers_path, diffusers_path) where each is Path to src/transformers or src/diffusers
    """
    # Find pyproject.toml
    current = Path.cwd()
    pyproject_path = None
    for parent in [current, *current.parents]:
        candidate = parent / "pyproject.toml"
        if candidate.exists():
            pyproject_path = candidate
            break

    if not pyproject_path:
        # Fallback to default Desktop paths
        desktop = Path.home() / "Desktop"
        transformers_repo = desktop / "transformers"
        diffusers_repo = desktop / "diffusers"
        transformers_path = transformers_repo / "src" / "transformers" if transformers_repo.exists() else None
        diffusers_path = diffusers_repo / "src" / "diffusers" if diffusers_repo.exists() else None
        return transformers_path, diffusers_path

    # Try to parse pyproject.toml
    try:
        try:
            import tomllib  # noqa: PLC0415

            toml_lib = tomllib
        except ImportError:
            try:
                import tomli  # noqa: PLC0415

                toml_lib = tomli  # type: ignore[assignment]
            except ImportError:
                # Fallback to manual parsing
                return _parse_pyproject_manually(pyproject_path)

        with open(pyproject_path, "rb") as f:
            pyproject = toml_lib.load(f)

        # Check tool.uv.sources for editable installs
        sources = pyproject.get("tool", {}).get("uv", {}).get("sources", {})
        transformers_repo = None
        diffusers_repo = None

        for name, config in sources.items():
            if isinstance(config, dict) and config.get("editable") and config.get("path"):
                abs_path = (pyproject_path.parent / config["path"]).resolve()
                if name == "transformers" and abs_path.exists():
                    transformers_repo = abs_path
                elif name == "diffusers" and abs_path.exists():
                    diffusers_repo = abs_path

        transformers_path = (transformers_repo / "src" / "transformers") if transformers_repo else None
        diffusers_path = (diffusers_repo / "src" / "diffusers") if diffusers_repo else None

        # Fallback to default Desktop paths if not found
        if not transformers_path:
            desktop = Path.home() / "Desktop"
            transformers_repo = desktop / "transformers"
            if transformers_repo.exists():
                transformers_path = transformers_repo / "src" / "transformers"

        if not diffusers_path:
            desktop = Path.home() / "Desktop"
            diffusers_repo = desktop / "diffusers"
            if diffusers_repo.exists():
                diffusers_path = diffusers_repo / "src" / "diffusers"

        return transformers_path, diffusers_path

    except Exception:  # noqa: BLE001
        return _parse_pyproject_manually(pyproject_path)


def _parse_pyproject_manually(pyproject_path: Path) -> tuple[Optional[Path], Optional[Path]]:
    """Manually parse pyproject.toml for editable paths."""
    import re

    try:
        with open(pyproject_path, encoding="utf-8") as f:
            content = f.read()

        transformers_repo = None
        diffusers_repo = None

        for line in content.split("\n"):
            if "transformers" in line and "path" in line and "editable" in line:
                match = re.search(r'path\s*=\s*"([^"]+)"', line)
                if match:
                    path_str = match.group(1)
                    abs_path = (pyproject_path.parent / path_str).resolve()
                    if abs_path.exists():
                        transformers_repo = abs_path

            if "diffusers" in line and "path" in line and "editable" in line:
                match = re.search(r'path\s*=\s*"([^"]+)"', line)
                if match:
                    path_str = match.group(1)
                    abs_path = (pyproject_path.parent / path_str).resolve()
                    if abs_path.exists():
                        diffusers_repo = abs_path

        transformers_path = (transformers_repo / "src" / "transformers") if transformers_repo else None
        diffusers_path = (diffusers_repo / "src" / "diffusers") if diffusers_repo else None

        # Fallback to default Desktop paths
        if not transformers_path:
            desktop = Path.home() / "Desktop"
            transformers_repo = desktop / "transformers"
            if transformers_repo.exists():
                transformers_path = transformers_repo / "src" / "transformers"

        if not diffusers_path:
            desktop = Path.home() / "Desktop"
            diffusers_repo = desktop / "diffusers"
            if diffusers_repo.exists():
                diffusers_path = diffusers_repo / "src" / "diffusers"

        return transformers_path, diffusers_path

    except Exception:  # noqa: BLE001
        desktop = Path.home() / "Desktop"
        transformers_repo = desktop / "transformers"
        diffusers_repo = desktop / "diffusers"
        transformers_path = transformers_repo / "src" / "transformers" if transformers_repo.exists() else None
        diffusers_path = diffusers_repo / "src" / "diffusers" if diffusers_repo.exists() else None
        return transformers_path, diffusers_path


def ensure_pruned_branch(repo_path: Path, repo_name: str) -> bool:
    """
    Ensure repo is on main-pruned branch, creating it if needed.

    Args:
        repo_path: Path to the repo root (e.g., ~/Desktop/transformers)
        repo_name: Name of repo for error messages

    Returns:
        True if successful, False otherwise
    """
    if not repo_path.exists():
        print(f"❌ Repo not found: {repo_path}")
        return False

    if not (repo_path / ".git").exists():
        print(f"❌ Not a git repo: {repo_path}")
        return False

    # Check current branch
    result = subprocess.run(
        ["git", "rev-parse", "--abbrev-ref", "HEAD"],
        cwd=repo_path,
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        print(f"❌ Failed to get current branch for {repo_name}")
        return False

    current_branch = result.stdout.strip()

    if current_branch == "main-pruned":
        print(f"✅ {repo_name}: Already on main-pruned branch")
        return True

    # Check if main-pruned branch exists
    result = subprocess.run(
        ["git", "branch", "--list", "main-pruned"],
        cwd=repo_path,
        capture_output=True,
        text=True,
    )

    branch_exists = bool(result.stdout.strip())

    if branch_exists:
        # Switch to existing branch
        result = subprocess.run(
            ["git", "checkout", "main-pruned"],
            cwd=repo_path,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            print(f"❌ Failed to switch to main-pruned branch for {repo_name}")
            print(f"   Error: {result.stderr}")
            return False
        print(f"✅ {repo_name}: Switched to main-pruned branch")
    else:
        # Create new branch from main
        result = subprocess.run(
            ["git", "checkout", "-b", "main-pruned"],
            cwd=repo_path,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            print(f"❌ Failed to create main-pruned branch for {repo_name}")
            print(f"   Error: {result.stderr}")
            return False
        print(f"✅ {repo_name}: Created and switched to main-pruned branch")

    return True


def git_commit_deletions(repo_path: Path, repo_name: str, script_name: str) -> bool:
    """
    Commit file deletions to git.

    Args:
        repo_path: Path to the repo root
        repo_name: Name of repo for messages
        script_name: Name of script being pruned

    Returns:
        True if successful, False otherwise
    """
    # Check if there are any changes
    result = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=repo_path,
        capture_output=True,
        text=True,
    )

    if not result.stdout.strip():
        print(f"   {repo_name}: No changes to commit")
        return True

    # Stage deletions
    result = subprocess.run(
        ["git", "add", "-A"],
        cwd=repo_path,
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        print(f"❌ Failed to stage changes for {repo_name}")
        return False

    # Commit
    commit_message = f"Prune: Remove unused files from {script_name}"
    result = subprocess.run(
        ["git", "commit", "-m", commit_message],
        cwd=repo_path,
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        print(f"❌ Failed to commit changes for {repo_name}")
        print(f"   Error: {result.stderr}")
        return False

    print(f"✅ {repo_name}: Committed deletions")
    return True


def profile_script(
    script_path: Path, transformers_repo: Optional[Path], diffusers_repo: Optional[Path]
) -> Optional[Path]:
    """
    Profile a script by injecting profiler start/stop around the main execution.

    Args:
        script_path: Path to script to profile
        transformers_repo: Path to transformers repo root (for filter_paths)
        diffusers_repo: Path to diffusers repo root (for filter_paths)

    Returns:
        Path to generated profile JSON file, or None if failed
    """
    print("=" * 70)
    print("🎯 PROFILING SCRIPT")
    print("=" * 70)
    print(f"Script: {script_path}")
    print()

    # Read the original script
    with open(script_path) as f:
        script_content = f.read()

    # Build filter paths
    filter_paths = []
    if transformers_repo:
        filter_paths.append(str(transformers_repo))
    if diffusers_repo:
        filter_paths.append(str(diffusers_repo))

    if not filter_paths:
        print("⚠️  Warning: No transformers/diffusers repos detected, profiling may be incomplete")

    # Create profiler setup code
    filter_paths_str = ",\n        ".join([f'"{path}"' for path in filter_paths])
    profiler_setup = f"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent if "examples" in __file__ else Path.cwd() / "src"))

from mflux_debugger.profiler_service import ExecutionProfiler
from datetime import datetime

# Create profiler
__profiler = ExecutionProfiler(
    filter_paths=[
        {filter_paths_str}
    ],
    max_depth=100,
    exclude_patterns=["__pycache__", "test_", "tests/"]
)

# Wrap the main execution
__timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
__profiler.start(session_id=f"profile_{{__timestamp}}", script_path=__file__)

import atexit

def __save_profile():
    __profiler.stop()
    __output_dir = Path("mflux_debugger/profiles")
    __output_dir.mkdir(parents=True, exist_ok=True)
    __output_file = __profiler.save(__output_dir)
    print(f"\\n✅ Profile saved: {{__output_file}}")
    # Write the path to a temp file so we can read it
    with open("/tmp/mflux_last_profile.txt", "w") as f:
        f.write(str(__output_file))

atexit.register(__save_profile)
"""

    # Find where to inject (after imports, before main execution)
    lines = script_content.split("\n")

    # Find the last import and inject after it
    last_import_idx = 0
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith(("import ", "from ")) and "atexit" not in stripped:
            last_import_idx = i

    # Insert profiler setup after last import
    modified_lines = lines[: last_import_idx + 1] + [profiler_setup] + lines[last_import_idx + 1 :]

    # Write modified script to temp file
    temp_script = Path(tempfile.gettempdir()) / f"mflux_profiled_{datetime.now().strftime('%Y%m%d_%H%M%S')}.py"
    with open(temp_script, "w") as f:
        f.write("\n".join(modified_lines))

    print("📦 Running script with profiling...")
    print()

    # Run the modified script
    result = subprocess.run(
        ["uv", "run", "python", str(temp_script)],
        cwd=Path.cwd(),
        capture_output=False,
    )

    # Clean up temp script
    try:
        temp_script.unlink()
    except Exception:  # noqa: BLE001
        pass

    if result.returncode != 0:
        print(f"\n⚠️  Script exited with code {result.returncode}")

    # Read the profile path from temp file
    profile_path_file = Path("/tmp/mflux_last_profile.txt")
    if profile_path_file.exists():
        profile_path = Path(profile_path_file.read_text().strip())
        if profile_path.exists():
            return profile_path

    # Fallback: find most recent profile
    profile_dir = Path("mflux_debugger/profiles")
    if profile_dir.exists():
        profiles = sorted(profile_dir.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
        if profiles:
            return profiles[0]

    return None


def check_editable_installs() -> bool:
    """Check if transformers and diffusers are installed in editable mode."""
    try:
        result = subprocess.run(
            ["uv", "pip", "list", "--editable"],
            capture_output=True,
            text=True,
            check=False,
        )

        if result.returncode != 0:
            print(f"⚠️  Warning: Could not check editable installs: {result.stderr}")
            return True  # Don't block

        editable_packages = result.stdout.lower()
        issues = []

        # Check if diffusers is editable
        if "diffusers" not in editable_packages:
            issues.append("⚠️  diffusers is NOT installed in editable mode")

        # Check if transformers is editable
        if "transformers" not in editable_packages:
            issues.append("⚠️  transformers is NOT installed in editable mode")

        if issues:
            print("\n" + "=" * 70)
            print("🚨 EDITABLE INSTALL CHECK FAILED")
            print("=" * 70)
            for issue in issues:
                print(f"   {issue}")
            print("\n💡 Why this matters:")
            print("   Pruning requires direct access to source files.")
            print("   Without editable installs, the tool cannot modify the repos.\n")
            print("💡 To fix:")
            print("   1. Clone libraries to a local directory:")
            print("      cd ~/Desktop  # or any directory")
            print("      git clone https://github.com/huggingface/transformers.git")
            print("      git clone https://github.com/huggingface/diffusers.git")
            print("   2. Install in editable mode:")
            print("      cd /path/to/mflux")
            print("      uv pip install -e /path/to/transformers")
            print("      uv pip install -e /path/to/diffusers")
            print("   3. Run this command again\n")
            print("=" * 70)
            return False

        # Success! Show where they're installed
        try:
            import diffusers  # noqa: PLC0415
            import transformers  # noqa: PLC0415

            print("✅ Editable installs verified:")
            print(f"   diffusers: {diffusers.__file__}")
            print(f"   transformers: {transformers.__file__}\n")
        except ImportError:
            print("✅ Editable installs configured (libraries not imported yet)\n")

        return True

    except Exception as e:  # noqa: BLE001
        print(f"⚠️  Warning: Could not verify editable installs: {e}")
        return True  # Don't block on unexpected errors


def cmd_setup() -> int:
    """Setup main-pruned branches in transformers/diffusers repos."""
    print("=" * 70)
    print("🔧 SETUP MAIN-PRUNED BRANCHES")
    print("=" * 70)
    print()

    # Check editable installs first
    if not check_editable_installs():
        print("\n❌ Cannot setup without proper editable installs")
        return 1

    transformers_path, diffusers_path = detect_repo_paths()

    success = True

    if transformers_path:
        transformers_repo = transformers_path.parent.parent
        if not ensure_pruned_branch(transformers_repo, "transformers"):
            success = False
    else:
        print("⚠️  Transformers repo not found")

    if diffusers_path:
        diffusers_repo = diffusers_path.parent.parent
        if not ensure_pruned_branch(diffusers_repo, "diffusers"):
            success = False
    else:
        print("⚠️  Diffusers repo not found")

    print()
    if success:
        print("✅ Setup complete!")
        return 0
    else:
        print("❌ Setup failed - check errors above")
        return 1


def cmd_prune(script_path: Path) -> int:
    """Profile script and prune unused files."""
    if not script_path.exists():
        print(f"❌ Script not found: {script_path}")
        return 1

    # Check editable installs first
    if not check_editable_installs():
        print("\n❌ Cannot prune without proper editable installs")
        return 1

    # Detect repo paths
    transformers_path, diffusers_path = detect_repo_paths()

    if not transformers_path and not diffusers_path:
        print("❌ Could not detect transformers or diffusers repos")
        print("   Make sure they're installed in editable mode or located at ~/Desktop/")
        return 1

    # Get repo roots
    transformers_repo = transformers_path.parent.parent if transformers_path else None
    diffusers_repo = diffusers_path.parent.parent if diffusers_path else None

    # Ensure we're on main-pruned branches
    print("=" * 70)
    print("🔧 ENSURING MAIN-PRUNED BRANCHES")
    print("=" * 70)
    print()

    if transformers_repo:
        if not ensure_pruned_branch(transformers_repo, "transformers"):
            return 1

    if diffusers_repo:
        if not ensure_pruned_branch(diffusers_repo, "diffusers"):
            return 1

    # Profile the script
    print()
    profile_path = profile_script(script_path, transformers_repo, diffusers_repo)

    if not profile_path:
        print("❌ Failed to generate profile")
        return 1

    # Analyze the profile
    print()
    print("=" * 70)
    print("📊 ANALYZING PROFILE")
    print("=" * 70)
    print()

    analysis = analyze_profile(profile_path)

    # Generate markdown report
    report_path = Path(f"PROFILE_REPORT_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md")
    generate_markdown_report(analysis, report_path, script_path)

    print()
    print("=" * 70)
    print("✅ REPORT GENERATED")
    print("=" * 70)
    print(f"📄 {report_path}")
    print()
    print("Files executed:")
    print(f"  • Transformers: {len(analysis['transformers_files'])}")
    print(f"  • Diffusers: {len(analysis['diffusers_files'])}")
    print()

    # Prune files
    print()
    print("=" * 70)
    print("🔪 PRUNING NON-EXECUTED FILES")
    print("=" * 70)
    print()

    script_name = script_path.stem

    # Prune files (function handles both repos)
    # Use dummy paths that don't exist if repos aren't found (pruner will skip them)
    dummy_path = Path("/nonexistent")
    kept_count, deleted_count, essential_kept = prune_files(
        analysis,
        transformers_path or dummy_path,
        diffusers_path or dummy_path,
        script_name,
    )
    print()
    print("Summary:")
    print(f"  ✅ Kept: {kept_count} files ({essential_kept} essential)")
    print(f"  ❌ Deleted: {deleted_count} files")
    if kept_count + deleted_count > 0:
        print(f"  📉 Reduction: {(deleted_count / (kept_count + deleted_count) * 100):.1f}%")

    # Commit deletions for each repo
    if transformers_repo:
        git_commit_deletions(transformers_repo, "transformers", script_name)

    if diffusers_repo:
        git_commit_deletions(diffusers_repo, "diffusers", script_name)

    print()
    print("=" * 70)
    print("✅ DONE")
    print("=" * 70)

    return 0


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        prog="mflux-debug-prune",
        description="Profile scripts and prune unused files from transformers/diffusers repos",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Setup main-pruned branches (one-time)
  mflux-debug-prune setup

  # Profile and prune a script
  mflux-debug-prune prune src/mflux_debugger/_scripts/debug_diffusers_txt2img.py

  # If script breaks after pruning, restore files via git:
  cd ~/Desktop/transformers
   git checkout HEAD~1 -- <file_path>
   git commit -m "Restore: <file_path>"

   # Then run prune again - it will keep the restored file
  mflux-debug-prune prune src/mflux_debugger/_scripts/debug_diffusers_txt2img.py
        """,
    )

    subparsers = parser.add_subparsers(dest="command", help="Command to execute")

    # Setup command
    subparsers.add_parser("setup", help="Setup main-pruned branches in transformers/diffusers repos")

    # Prune command
    prune_parser = subparsers.add_parser("prune", help="Profile script and prune unused files")
    prune_parser.add_argument("script", type=Path, help="Path to script to profile")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return 1

    if args.command == "setup":
        return cmd_setup()
    elif args.command == "prune":
        return cmd_prune(args.script)
    else:
        parser.print_help()
        return 1


if __name__ == "__main__":
    sys.exit(main())
