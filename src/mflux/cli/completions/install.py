#!/usr/bin/env python3
import argparse
import os
import subprocess
import sys
from pathlib import Path

from mflux.cli.completions.generator import CompletionGenerator


def get_zsh_fpath():
    try:
        result = subprocess.run(
            ["zsh", "-c", "echo $fpath"],
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip().split()
    except subprocess.CalledProcessError:
        return []


def find_completion_dir():
    # Common completion directories in order of preference
    candidates = [
        Path.home() / ".zsh" / "completions",
        Path.home() / ".config" / "zsh" / "completions",
        Path("/usr/local/share/zsh/site-functions"),
        Path("/opt/homebrew/share/zsh/site-functions"),
    ]

    # Check fpath directories
    fpath = get_zsh_fpath()
    for path_str in fpath:
        path = Path(path_str)
        if path.exists() and os.access(path, os.W_OK):
            candidates.insert(0, path)

    # Return first writable directory
    for candidate in candidates:
        if candidate.exists() and os.access(candidate, os.W_OK):
            return candidate

    # Default to user directory
    user_dir = Path.home() / ".zsh" / "completions"
    user_dir.mkdir(parents=True, exist_ok=True)
    return user_dir


def check_installation():
    print("Checking mflux completions installation...\n")

    # Check if completion file exists
    found_locations = []
    fpath = get_zsh_fpath()

    for path_str in fpath:
        path = Path(path_str)
        completion_file = path / "_mflux"
        if completion_file.exists():
            found_locations.append(completion_file)

    if not found_locations:
        print("❌ No mflux completion file found in $fpath")
        print("\nSearched in:")
        for path_str in fpath[:5]:  # Show first 5 locations
            print(f"  - {path_str}")
        print("\nRun 'mflux-completions' to install completions")
        return False

    print("✓ Found completion file(s):")
    for location in found_locations:
        print(f"  - {location}")

    # Check if compinit is loaded in interactive shell
    try:
        result = subprocess.run(
            ["zsh", "-i", "-c", "type compinit"],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode == 0:
            print("\n✓ compinit is available")
        else:
            print("\n⚠️  compinit may not be initialized")
            print("   If completions don't work, add to ~/.zshrc:")
            print("   autoload -U compinit && compinit")
    except (subprocess.SubprocessError, OSError):
        print("\n⚠️  Could not verify compinit")

    # Test a completion
    print("\n✓ Installation appears correct!")
    print("\nTo test completions:")
    print("  1. Start a new shell or run: exec zsh")
    print("  2. Type: mflux-generate --<TAB>")
    print("\nIf completions don't work, try:")
    print("  rm -f ~/.zcompdump && compinit")

    return True


def main():
    parser = argparse.ArgumentParser(
        description="Install or generate ZSH completions for mflux commands",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Install to default location
  mflux-completions

  # Generate to stdout
  mflux-completions --print

  # Install to specific directory
  mflux-completions --dir ~/.config/zsh/completions

  # Update existing completion
  mflux-completions --update

  # Check if completions are properly installed
  mflux-completions --check
""",
    )

    parser.add_argument(
        "--print",
        action="store_true",
        help="Print completion script to stdout instead of installing",
    )
    parser.add_argument(
        "--dir",
        type=Path,
        help="Directory to install completion file (default: auto-detect)",
    )
    parser.add_argument(
        "--update",
        action="store_true",
        help="Update existing completion file",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Check if completions are properly installed",
    )

    args = parser.parse_args()

    # Handle --check flag
    if args.check:
        sys.exit(0 if check_installation() else 1)

    # Generate completion script
    generator = CompletionGenerator()
    completion_script = generator.generate()

    if args.print:
        # Just print to stdout
        print(completion_script)
        return

    # Determine installation directory
    if args.dir:
        install_dir = args.dir
        install_dir.mkdir(parents=True, exist_ok=True)
    else:
        install_dir = find_completion_dir()

    # Install completion file
    completion_file = install_dir / "_mflux"

    if completion_file.exists() and not args.update:
        print(f"Error: Completion file already exists at {completion_file}")
        print("Use --update to overwrite")
        sys.exit(1)

    try:
        completion_file.write_text(completion_script)
        print(f"✓ Installed ZSH completions to: {completion_file}")
        print()
        print("To activate completions:")
        print("1. Ensure this directory is in your $fpath:")
        print(f"   echo 'fpath=({install_dir} $fpath)' >> ~/.zshrc")
        print("2. Reload your shell:")
        print("   exec zsh")
        print()
        print("If completions don't work immediately, you may need to:")
        print("   rm -f ~/.zcompdump && compinit")

    except OSError as e:
        print(f"Error installing completion file: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
