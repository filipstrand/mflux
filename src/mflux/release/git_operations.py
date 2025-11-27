import subprocess

from mflux.utils.exceptions import CommandExecutionError


class GitOperations:
    @staticmethod
    def run_command(cmd: list, description: str, check: bool = True) -> subprocess.CompletedProcess:
        print(f"🔄 {description}...")
        print(f"   Running: {' '.join(cmd)}")

        result = subprocess.run(cmd, capture_output=True, text=True, check=False)

        if result.returncode != 0 and check:
            print(f"❌ {description} failed!")
            print(f"   stdout: {result.stdout}")
            print(f"   stderr: {result.stderr}")
            raise CommandExecutionError(cmd, result.returncode, result.stdout, result.stderr)

        if result.stdout:
            print(f"   stdout: {result.stdout.strip()}")

        print(f"✅ {description} completed successfully")
        return result

    @staticmethod
    def check_tag_exists(tag_name: str) -> bool:
        print("🔍 Checking git tag existence...")
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

            print(f"   Git tag {tag_name} does not exist")
            return False
        except subprocess.CalledProcessError as e:
            print(f"⚠️  Warning: Could not check git tag existence: {e}")
            return False

    @staticmethod
    def create_and_push_tag(tag_name: str, version: str) -> None:
        print("🏷️  Creating git tag...")
        GitOperations.run_command(
            ["git", "tag", "-a", tag_name, "-m", f"Release {version}"], f"Creating git tag {tag_name}"
        )
        GitOperations.run_command(["git", "push", "origin", tag_name], f"Pushing git tag {tag_name} to remote")
