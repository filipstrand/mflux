import getpass
import hashlib
import os
import shutil
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
        repo_cache.mkdir(parents=True, exist_ok=True)
        (repo_cache / HfCacheAlias.MARKER).write_text(f"{model_dir}\n")
        # Build outside snapshots/, because the loader examines each folder in it. Then swap the new
        # snapshot in, so a reader never sees a half-built snapshot and the old one stays until the end.
        building = repo_cache / f".mflux_building-{revision}"
        replaced = repo_cache / f".mflux_replaced-{revision}"
        for leftover in (building, replaced):
            if leftover.exists():
                shutil.rmtree(leftover)
        building.mkdir()
        file_count = 0
        for source in HfCacheAlias._model_files(model_dir):
            target = building / source.relative_to(model_dir)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.symlink_to(source)
            file_count += 1
        snapshot = repo_cache / "snapshots" / revision
        snapshot.parent.mkdir(exist_ok=True)
        if snapshot.exists():
            snapshot.rename(replaced)
        building.rename(snapshot)
        if replaced.exists():
            shutil.rmtree(replaced)
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
