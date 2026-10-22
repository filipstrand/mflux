import json
import os
import struct
import tempfile
from contextlib import contextmanager
from pathlib import Path

import mlx.core as mx
from huggingface_hub import HfFileSystem, constants, snapshot_download
from huggingface_hub.errors import GatedRepoError, HfHubHTTPError, RepositoryNotFoundError
from huggingface_hub.utils import hf_raise_for_status, http_stream_backoff
from tqdm import tqdm

from mflux.cli.defaults.defaults import MFLUX_CACHE_DIR
from mflux.models.common.resolution.model_revisions import ModelRevisions


@contextmanager
def _open_range(filesystem: HfFileSystem, path: str, start: int, end: int):
    """Open a validated HTTP range connection to the bytes [start, end) of `path`.

    The server must honor the range request. A 200 (full body) response would make
    the bytes misaligned, so we require HTTP 206 and a Content-Range that starts at
    `start` before yielding the response.
    """
    url = filesystem.url(path)
    headers = {"range": f"bytes={start}-{end - 1}", **filesystem._api._build_hf_headers()}
    with http_stream_backoff("GET", url, headers=headers, timeout=constants.HF_HUB_DOWNLOAD_TIMEOUT) as response:
        hf_raise_for_status(response)
        if response.status_code != 206:
            raise OSError(f"Expected HTTP 206 for the range read but got {response.status_code}.")
        content_range = response.headers.get("content-range", "")
        if not content_range.startswith(f"bytes {start}-"):
            raise OSError(f"Content-Range '{content_range}' does not start at the requested offset {start}.")
        yield response


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
        # The data_offsets are relative to the first byte after the header, so the HTTP range is file-absolute.
        file_start, file_end = 8 + header_size + start, 8 + header_size + end
        destination.parent.mkdir(parents=True, exist_ok=True)
        # A file unique to this process, so two concurrent downloads never write into the same partial file.
        handle, name = tempfile.mkstemp(dir=destination.parent, prefix=f"{destination.name}.", suffix=".partial")
        partial = Path(name)
        try:
            with os.fdopen(handle, "wb") as local:
                local.write(struct.pack("<Q", len(encoded)) + encoded)
                with _open_range(filesystem, path, file_start, file_end) as response:
                    with tqdm(total=end - start, unit="B", unit_scale=True, desc="zen-image-edit text_fusion") as bar:
                        remaining = end - start
                        for chunk in response.iter_bytes():
                            if remaining <= 0:
                                break
                            to_write = chunk[:remaining]
                            if not to_write:
                                continue
                            local.write(to_write)
                            remaining -= len(to_write)
                            bar.update(len(to_write))
                        if remaining:
                            raise OSError("The text_fusion download ended early.")
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
