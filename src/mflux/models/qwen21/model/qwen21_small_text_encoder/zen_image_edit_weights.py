import json
import os
import struct
import tempfile
from pathlib import Path

import mlx.core as mx
from huggingface_hub import HfFileSystem, snapshot_download
from huggingface_hub.errors import GatedRepoError, HfHubHTTPError, RepositoryNotFoundError
from tqdm import tqdm

from mflux.cli.defaults.defaults import MFLUX_CACHE_DIR
from mflux.models.common.resolution.model_revisions import ModelRevisions


class ZenImageEditWeights:
    REPO = "AiArtLab/zen-image-edit"
    FUSION_PREFIX = "text_fusion."
    SNAPSHOT_PATTERNS = [
        "processor/*",
        "text_encoder/config.json",
        "text_encoder/model.safetensors",
        "transformer/config.json",
        "transformer/diffusion_pytorch_model.safetensors.index.json",
    ]
    _CHUNK_BYTES = 16 * 1024 * 1024

    @staticmethod
    def snapshot() -> Path:
        return Path(
            ZenImageEditWeights._friendly(
                lambda: snapshot_download(
                    ZenImageEditWeights.REPO,
                    revision=ZenImageEditWeights.revision(),
                    allow_patterns=ZenImageEditWeights.SNAPSHOT_PATTERNS,
                )
            )
        )

    @staticmethod
    def revision() -> str:
        return ModelRevisions.get(ZenImageEditWeights.REPO) or "main"

    @staticmethod
    def text_fusion_weights(snapshot: Path) -> dict[str, mx.array]:
        # The 158M adapter sits inside the 14.5 GB fp16 DiT. mflux runs the Qwen-Image-2.1 DiT, so only the
        # text_fusion.* byte range is fetched (one HTTP range request) and cached as its own file.
        cached = MFLUX_CACHE_DIR / "zen-image-edit" / ZenImageEditWeights.revision() / "text_fusion.safetensors"
        if not cached.is_file():
            index = json.loads((snapshot / "transformer/diffusion_pytorch_model.safetensors.index.json").read_text())
            shards = {
                shard for key, shard in index["weight_map"].items() if key.startswith(ZenImageEditWeights.FUSION_PREFIX)
            }
            if len(shards) != 1:
                raise ValueError(f"Expected the text_fusion weights in one shard of {ZenImageEditWeights.REPO}.")
            ZenImageEditWeights._friendly(lambda: ZenImageEditWeights._extract(f"transformer/{shards.pop()}", cached))
        weights = mx.load(str(cached))
        return {key.removeprefix(ZenImageEditWeights.FUSION_PREFIX): value for key, value in weights.items()}

    @staticmethod
    def _extract(filename: str, destination: Path) -> None:
        path = f"{ZenImageEditWeights.REPO}@{ZenImageEditWeights.revision()}/{filename}"
        filesystem = HfFileSystem()
        with filesystem.open(path, "rb", block_size=0) as remote:
            header_size = struct.unpack("<Q", remote.read(8))[0]
            header = json.loads(remote.read(header_size))
        tensors = {key: entry for key, entry in header.items() if key.startswith(ZenImageEditWeights.FUSION_PREFIX)}
        start = min(entry["data_offsets"][0] for entry in tensors.values())
        end = max(entry["data_offsets"][1] for entry in tensors.values())
        new_header = {
            key: {**entry, "data_offsets": [entry["data_offsets"][0] - start, entry["data_offsets"][1] - start]}
            for key, entry in tensors.items()
        }
        encoded = json.dumps(new_header, separators=(",", ":")).encode()
        encoded += b" " * (-len(encoded) % 8)
        destination.parent.mkdir(parents=True, exist_ok=True)
        # A file unique to this process, so two concurrent downloads never write into the same partial file.
        handle, name = tempfile.mkstemp(dir=destination.parent, prefix=f"{destination.name}.", suffix=".partial")
        partial = Path(name)
        block = ZenImageEditWeights._CHUNK_BYTES
        try:
            with filesystem.open(path, "rb", block_size=block) as remote, os.fdopen(handle, "wb") as local:
                local.write(struct.pack("<Q", len(encoded)) + encoded)
                remote.seek(8 + header_size + start)
                with tqdm(total=end - start, unit="B", unit_scale=True, desc="zen-image-edit text_fusion") as bar:
                    remaining = end - start
                    while remaining:
                        chunk = remote.read(min(remaining, block))
                        if not chunk:
                            raise OSError("The text_fusion download ended early.")
                        local.write(chunk)
                        remaining -= len(chunk)
                        bar.update(len(chunk))
            partial.replace(destination)
        except BaseException:
            partial.unlink(missing_ok=True)
            raise

    @staticmethod
    def _friendly(fn):
        try:
            return fn()
        except (GatedRepoError, RepositoryNotFoundError) as e:
            raise RuntimeError(
                f"--small-text-encoder: Hugging Face repo '{ZenImageEditWeights.REPO}' was not found or is not accessible."
            ) from e
        except (HfHubHTTPError, OSError) as e:
            raise RuntimeError(
                f"--small-text-encoder: downloading from '{ZenImageEditWeights.REPO}' failed ({e}). "
                "It needs about 2.1 GB (the Qwen3.5-0.8B encoder and the text_fusion adapter). "
                "Check your network connection or HF_HUB_OFFLINE setting and retry."
            ) from e
