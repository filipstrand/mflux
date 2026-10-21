import hashlib
import json
import os
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from huggingface_hub import (
    HfApi,
    constants as hf_constants,
)

from mflux.models.common.config.model_config import AVAILABLE_MODELS
from mflux.models.common.resolution.model_revisions import ModelRevisions
from mflux.utils.version_util import VersionUtil

GB = 1000**3


@dataclass(frozen=True)
class Upstream:
    repo_id: str | None
    snapshot: Path | None
    revision: str | None


class SaveTarget:
    # Picks the folder that mflux-save writes to: --path, else $MFLUX_MODELS_DIR/<auto name>.
    MODELS_DIR_ENV_VAR = "MFLUX_MODELS_DIR"
    # An earlier mflux save loaded as a local --model ("z-image-turbo-MFlux-q8") keeps only its base name.
    OLD_SUFFIX = re.compile(r"-mflux-(q\d+|bf16|fp8)(-lora\d+)?$", re.IGNORECASE)

    @staticmethod
    def resolve(path_arg: str | None, key: str, model_path: str | None, quantize: int | None, has_loras: bool) -> Path:
        if path_arg:
            return Path(path_arg).expanduser().resolve()
        models_dir_value = os.environ.get(SaveTarget.MODELS_DIR_ENV_VAR)
        if not models_dir_value:
            raise ValueError(f"no target folder. Pass --path, or set ${SaveTarget.MODELS_DIR_ENV_VAR}.")
        models_dir = Path(models_dir_value).expanduser().resolve()
        if not models_dir.is_dir():
            raise ValueError(f"${SaveTarget.MODELS_DIR_ENV_VAR} is not a folder: {models_dir}")
        if not os.access(models_dir, os.W_OK):
            raise ValueError(f"${SaveTarget.MODELS_DIR_ENV_VAR} is not writable: {models_dir}")

        name = f"{SaveTarget.base_name(key, model_path)}-mflux-{SaveTarget.precision_label(key, quantize)}"
        if has_loras:
            return models_dir / f"{name}-lora{SaveTarget._next_lora_number(models_dir, name):02d}"
        target = models_dir / name
        if target.exists():
            raise ValueError(f"{target} exists. Delete it, or pass --path to write to it.")
        return target

    @staticmethod
    def base_name(key: str, model_path: str | None) -> str:
        if model_path is None:
            return SaveTarget._registry_name(key)
        local = Path(model_path).expanduser()
        if local.exists():
            repo_id = UpstreamLocator.repo_id_from_cache_path(local.resolve())
            name = repo_id.split("/")[-1] if repo_id else local.resolve().name
        else:
            name = model_path.rstrip("/").split("/")[-1]
        # "models--org--name" and "org--name" folders: keep only the part after the last "--",
        # because the Hub refuses "--" in a repo name.
        name = name.split("--")[-1]
        return SaveTarget.OLD_SUFFIX.sub("", name)

    @staticmethod
    def stored_quantization(model_path: str | None) -> int | None:
        # The bit width of an earlier mflux save, from its index files. The loader keeps the stored
        # width and ignores -q (QuantizationResolution), so the name and the metadata must use it too.
        # A repo id ("bob/Z-Image-Turbo-mflux-q5") is read from its cached snapshot, as the loader does.
        if model_path is None:
            return None
        root = Path(model_path).expanduser()
        if not root.is_dir():
            root = UpstreamLocator.find(None, model_path).snapshot
            if root is None or not root.is_dir():
                return None
        for index in sorted([*root.glob("model.safetensors.index.json"), *root.glob("*/model.safetensors.index.json")]):
            try:
                level = json.loads(index.read_text()).get("metadata", {}).get("quantization_level")
                if level not in (None, "None", "null", ""):
                    return int(level)
            except (OSError, ValueError, TypeError, AttributeError):  # noqa: PERF203
                continue
        return None

    @staticmethod
    def precision_label(key: str, quantize: int | None) -> str:
        if quantize is not None:
            return f"q{quantize}"
        # Without --quantize, a model keeps its stored precision: FP8 for the FP8 checkpoints.
        return "fp8" if "fp8" in key else "bf16"

    @staticmethod
    def _registry_name(key: str) -> str:
        # The repo name. When several keys share one repo, add the part of the key that differs,
        # so two models never get one name: FLUX.1-dev for dev, FLUX.1-dev-controlnet-canny for
        # dev-controlnet-canny, SeedVR2_comfyUI-3b and SeedVR2_comfyUI-7b.
        repo_id = AVAILABLE_MODELS[key].model_name
        repo_name = repo_id.split("/")[-1]
        sharing = [k for k, config in AVAILABLE_MODELS.items() if config.model_name == repo_id]
        if len(sharing) == 1:
            return repo_name
        extra = key[len(os.path.commonprefix(sharing)) :].strip("-")
        return f"{repo_name}-{extra}" if extra else repo_name

    @staticmethod
    def _next_lora_number(models_dir: Path, name: str) -> int:
        pattern = re.compile(rf"^{re.escape(name)}-lora(\d+)$")
        numbers = [int(m.group(1)) for p in models_dir.iterdir() if (m := pattern.match(p.name))]
        return max(numbers, default=0) + 1


class UpstreamLocator:
    # Finds the upstream repo, revision and cached snapshot that a save loaded from. Reads the
    # local cache only: the one network call is the optional revision date.
    @staticmethod
    def find(model_config_repo_id: str | None, model_path: str | None) -> Upstream:
        if model_path is not None:
            local = Path(model_path).expanduser()
            if local.exists():
                local = local.resolve()
                repo_id = UpstreamLocator.repo_id_from_cache_path(local)
                if repo_id is None:
                    return Upstream(repo_id=None, snapshot=local, revision=None)
                if local.parent.name == "snapshots":
                    return Upstream(repo_id=repo_id, snapshot=local, revision=local.name)
                return UpstreamLocator._from_repo_cache(repo_id, local)
            repo_id = model_path
        else:
            repo_id = model_config_repo_id
        if repo_id is None:
            return Upstream(repo_id=None, snapshot=None, revision=None)
        repo_cache = Path(hf_constants.HF_HUB_CACHE) / f"models--{repo_id.replace('/', '--')}"
        return UpstreamLocator._from_repo_cache(repo_id, repo_cache)

    @staticmethod
    def repo_id_from_cache_path(path: Path) -> str | None:
        # ".../models--org--name" or ".../models--org--name/snapshots/<revision>"
        for part in (path, path.parent.parent):
            if part.name.startswith("models--") and part.name.count("--") >= 2:
                org, name = part.name[len("models--") :].split("--", 1)
                return f"{org}/{name}"
        return None

    @staticmethod
    def revision_date(repo_id: str | None, revision: str | None) -> str | None:
        if repo_id is None or revision is None or hf_constants.HF_HUB_OFFLINE:
            return None
        try:
            info = HfApi().model_info(repo_id, revision=revision, timeout=5)
        except Exception:  # noqa: BLE001 - the date is optional; any Hub failure leaves it null
            return None
        return SavedModelInfo.timestamp(info.last_modified) if info.last_modified else None

    @staticmethod
    def _from_repo_cache(repo_id: str, repo_cache: Path) -> Upstream:
        snapshots = repo_cache / "snapshots"
        if not snapshots.is_dir():
            return Upstream(repo_id=repo_id, snapshot=None, revision=None)
        # The same choice the loader makes: the pinned commit, else the newest snapshot.
        pinned = ModelRevisions.get(repo_id)
        if pinned is not None:
            snapshot = snapshots / pinned
        else:
            candidates = [p for p in snapshots.iterdir() if p.is_dir()]
            snapshot = max(candidates, key=lambda p: p.stat().st_mtime, default=None)
        if snapshot is None or not snapshot.is_dir():
            return Upstream(repo_id=repo_id, snapshot=None, revision=None)
        return Upstream(repo_id=repo_id, snapshot=snapshot, revision=snapshot.name)


class SavedModelInfo:
    # mflux_model.json: a readable record of what a save holds. The loader does not read it.
    FILE_NAME = "mflux_model.json"
    FORMAT_VERSION = 1
    TEXT_ENCODER_DIRS = ("text_encoder", "text_encoder_2", "mllm")

    @staticmethod
    def write(
        target: Path,
        key: str,
        model_path: str | None,
        model_config_repo_id: str | None,
        quantize: int | None,
        lora_sources: list[str],
        lora_files: list[str],
        lora_scales: list[float],
    ) -> dict:
        upstream = UpstreamLocator.find(model_config_repo_id, model_path)
        is_builtin = model_path is None
        # A local path inside an HF cache is named by its repo id, not by its "models--org--name" folder.
        model_name = key if is_builtin else (upstream.repo_id or SavedModelInfo._public_source(model_path))
        now = SavedModelInfo.timestamp(datetime.now(timezone.utc))
        created_at = SavedModelInfo._previous_created_at(target, model_name) or now
        info = {
            "format_version": SavedModelInfo.FORMAT_VERSION,
            "mflux_version": VersionUtil.get_mflux_version(),
            "created_at": created_at,
            "updated_at": now,
            "model_name": model_name,
            "base_model": None if is_builtin else key,
            "quantize": quantize,
            "size_gb": SavedModelInfo._size_gb(target),
            "upstream": None
            if upstream.repo_id is None and upstream.snapshot is None
            else {
                "repo_id": upstream.repo_id,
                "revision": upstream.revision,
                "revision_date": UpstreamLocator.revision_date(upstream.repo_id, upstream.revision),
                "size_gb": SavedModelInfo._size_gb(upstream.snapshot) if upstream.snapshot else None,
            },
            "text_encoders": SavedModelInfo._text_encoders(target, upstream.snapshot),
            "loras": [
                {
                    "source": SavedModelInfo._public_source(source),
                    "scale": scale,
                    "sha256": SavedModelInfo._sha256(Path(file)),
                }
                for source, file, scale in zip(lora_sources, lora_files, lora_scales)
            ],
        }
        # Write to a temporary name, then rename, so a crash never leaves half a JSON file.
        temporary = target / f".{SavedModelInfo.FILE_NAME}.tmp"
        temporary.write_text(json.dumps(info, indent=2) + "\n")
        temporary.replace(target / SavedModelInfo.FILE_NAME)
        return info

    @staticmethod
    def timestamp(moment: datetime) -> str:
        return moment.astimezone(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")

    @staticmethod
    def _public_source(source: str) -> str:
        # A local path keeps only its last part: the file can go to the Hub, and a full path names the user.
        local = Path(source).expanduser()
        return local.name if local.exists() else source

    @staticmethod
    def _previous_created_at(target: Path, model_name: str) -> str | None:
        try:
            previous = json.loads((target / SavedModelInfo.FILE_NAME).read_text())
        except (OSError, ValueError):
            return None
        return previous.get("created_at") if previous.get("model_name") == model_name else None

    @staticmethod
    def _size_gb(folder: Path) -> float:
        # Each file counts one time: snapshot entries are symlinks into a shared blob store.
        files = {p.resolve() for p in folder.rglob("*") if p.is_file() and p.name != SavedModelInfo.FILE_NAME}
        return round(sum(f.stat().st_size for f in files) / GB, 2)

    @staticmethod
    def _text_encoders(target: Path, snapshot: Path | None) -> list[str]:
        encoders = []
        for subdir in SavedModelInfo.TEXT_ENCODER_DIRS:
            if not (target / subdir).is_dir():
                continue
            model_type = None
            for root in (snapshot, target):
                if root is None:
                    continue
                try:
                    model_type = json.loads((root / subdir / "config.json").read_text()).get("model_type")
                except (OSError, ValueError):
                    continue
                if model_type:
                    break
            encoders.append(model_type or subdir)
        return encoders

    @staticmethod
    def _sha256(file: Path) -> str:
        digest = hashlib.sha256()
        with file.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1 << 20), b""):
                digest.update(chunk)
        return digest.hexdigest()
