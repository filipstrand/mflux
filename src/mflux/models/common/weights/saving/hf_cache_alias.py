import getpass
import hashlib
import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path

from huggingface_hub import constants as hf_constants
from huggingface_hub.utils import HFValidationError, validate_repo_id


@dataclass(frozen=True)
class HfCacheAliasResult:
    repo_id: str
    revision: str
    file_count: int
    repo_cache: Path


class HfCacheAlias:
    # Makes a saved model folder load as a Hugging Face repo id (<username>/<folder name>).
    # The snapshot holds one symlink per model file, so mflux copies no model data.
    MARKER = ".mflux_alias"
    # Highest priority first. The command-line name wins over all of these.
    USERNAME_ENV_VARS = ("MFLUX_HF_USERNAME", "HF_NAMESPACE", "HF_NAME")

    @staticmethod
    def resolve_username(cli_name: str | None) -> tuple[str, str]:
        if cli_name:
            return cli_name, "--hf-symlink"
        for env_var in HfCacheAlias.USERNAME_ENV_VARS:
            value = os.environ.get(env_var)
            if value:
                return value, env_var
        return getpass.getuser(), "OS user name"

    @staticmethod
    def check(model_dir: Path, username: str) -> None:
        # Runs before the save, so a bad name stops the command before any weights load.
        # The Hub's own rule: it also refuses names such as "model..8bit" and "model.git".
        repo_id = f"{username}/{model_dir.name}"
        try:
            validate_repo_id(repo_id)
        except HFValidationError as error:
            raise ValueError(
                f"'{repo_id}' is not a valid Hugging Face repo id. Use only letters, numbers, '.', '_' and '-'. "
                "Do not use '..', a '.git' end, or a '-' or '.' at the start or end."
            ) from error
        repo_cache = HfCacheAlias._repo_cache(model_dir, username)
        if repo_cache.exists() and not (repo_cache / HfCacheAlias.MARKER).exists():
            raise ValueError(f"{repo_cache} exists and mflux did not make it. mflux does not change it.")

    @staticmethod
    def create(model_dir: Path, username: str) -> HfCacheAliasResult:
        model_dir = model_dir.expanduser().resolve()
        HfCacheAlias.check(model_dir, username)
        repo_cache = HfCacheAlias._repo_cache(model_dir, username)
        revision = HfCacheAlias._revision(model_dir)
        # The marker goes first: if the command stops before the end, the next run can repair the entry.
        # A new entry gets its marker in a temporary folder that is then renamed, so the entry is never
        # without a marker. check() refuses an entry without one.
        umask = os.umask(0)
        os.umask(umask)
        if repo_cache.exists():
            (repo_cache / HfCacheAlias.MARKER).write_text(f"{model_dir}\n")
        else:
            repo_cache.parent.mkdir(parents=True, exist_ok=True)
            staging = Path(tempfile.mkdtemp(prefix=f".mflux_alias-{repo_cache.name}-", dir=repo_cache.parent))
            try:
                staging.chmod(0o777 & ~umask)
                (staging / HfCacheAlias.MARKER).write_text(f"{model_dir}\n")
                staging.rename(repo_cache)
            except BaseException:
                shutil.rmtree(staging, ignore_errors=True)
                raise
        snapshot = repo_cache / "snapshots" / revision
        snapshot.parent.mkdir(exist_ok=True)
        # An earlier run stopped between the two renames below: put its old snapshot back first.
        for leftover in sorted(repo_cache.glob(f".mflux_replaced-{revision}-*")):
            if snapshot.exists():
                shutil.rmtree(leftover, ignore_errors=True)
            else:
                leftover.rename(snapshot)
        # Build outside snapshots/, because the loader examines each folder in it. The folder name is
        # unique, so a run does not delete the build of another run. Runs at the same time are not
        # locked: the swap of one run can still fail if another run swaps at the same moment.
        building = Path(tempfile.mkdtemp(prefix=f".mflux_building-{revision}-", dir=repo_cache))
        # mkdtemp makes the folder 0700. A shared cache needs the mode that mkdir gives.
        building.chmod(0o777 & ~umask)
        try:
            file_count = 0
            for source in HfCacheAlias._model_files(model_dir):
                target = building / source.relative_to(model_dir)
                target.parent.mkdir(parents=True, exist_ok=True)
                target.symlink_to(source)
                file_count += 1
        except BaseException:
            shutil.rmtree(building, ignore_errors=True)
            raise
        # Swap the new snapshot in. The old snapshot stays until the new one is in place.
        replaced = Path(f"{building}".replace(".mflux_building-", ".mflux_replaced-", 1))
        if snapshot.exists():
            snapshot.rename(replaced)
        try:
            building.rename(snapshot)
        except BaseException:
            if replaced.exists():
                replaced.rename(snapshot)
            shutil.rmtree(building, ignore_errors=True)
            raise
        if replaced.exists():
            shutil.rmtree(replaced, ignore_errors=True)
        (repo_cache / "refs").mkdir(exist_ok=True)
        ref = repo_cache / "refs" / ".main.tmp"
        ref.write_text(revision)
        ref.replace(repo_cache / "refs" / "main")
        return HfCacheAliasResult(
            repo_id=f"{username}/{model_dir.name}",
            revision=revision,
            file_count=file_count,
            repo_cache=repo_cache,
        )

    @staticmethod
    def _repo_cache(model_dir: Path, username: str) -> Path:
        return Path(hf_constants.HF_HUB_CACHE).expanduser() / f"models--{username}--{model_dir.name}"

    @staticmethod
    def _revision(model_dir: Path) -> str:
        # 40 hex characters, the shape of a git commit hash. The Hub client does not check it.
        return hashlib.sha256(str(model_dir).encode()).hexdigest()[:40]

    @staticmethod
    def _model_files(model_dir: Path) -> list[Path]:
        files = []
        for root, dirs, names in os.walk(model_dir):
            dirs[:] = sorted(d for d in dirs if not d.startswith("."))
            files.extend(Path(root) / name for name in sorted(names) if not name.startswith("."))
        return files
