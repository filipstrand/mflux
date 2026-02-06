from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mlx.core as mx


@dataclass(frozen=True)
class CachePaths:
    root: Path
    data: Path


class TrainingDataCache:
    CACHE_DIRNAME = ".mflux_cache/training"
    CACHE_SCHEMA_VERSION = 1

    @staticmethod
    def get_paths(*, data_root: Path) -> CachePaths:
        root = data_root / TrainingDataCache.CACHE_DIRNAME
        return CachePaths(root=root, data=root / "data")

    @staticmethod
    def wipe_and_init(*, data_root: Path) -> CachePaths:
        paths = TrainingDataCache.get_paths(data_root=data_root)
        if paths.root.exists():
            shutil.rmtree(paths.root)
        paths.data.mkdir(parents=True, exist_ok=True)
        return paths

    @staticmethod
    def save_item(
        *,
        paths: CachePaths,
        data_id: int,
        prompt: str,
        image_path: Path,
        width: int,
        height: int,
        clean_latents: mx.array,
        cond: Any,
    ) -> None:
        tensor_path = paths.data / f"{data_id:07d}.safetensors"
        meta_path = paths.data / f"{data_id:07d}.json"

        tensors: dict[str, mx.array] = {
            "clean_latents": clean_latents,
            "width": mx.array(width, dtype=mx.int32),
            "height": mx.array(height, dtype=mx.int32),
        }
        if isinstance(cond, mx.array):
            tensors["cond"] = cond
        elif isinstance(cond, dict):
            for k, v in cond.items():
                if not isinstance(k, str) or not isinstance(v, mx.array):
                    raise TypeError("TrainingDataCache only supports cond as mx.array or dict[str, mx.array]")
                tensors[f"cond__{k}"] = v
        else:
            raise TypeError("TrainingDataCache only supports cond as mx.array or dict[str, mx.array]")

        # Atomic-ish write: write to temp and rename.
        # Note: mx.save_safetensors expects a `.safetensors` filename.
        tmp_tensor = tensor_path.with_name(f"{tensor_path.stem}.tmp.safetensors")
        mx.save_safetensors(
            str(tmp_tensor),
            tensors,
            metadata={"mflux_cache_schema": str(TrainingDataCache.CACHE_SCHEMA_VERSION)},
        )
        tmp_tensor.replace(tensor_path)

        meta = {
            "schema": TrainingDataCache.CACHE_SCHEMA_VERSION,
            "data_id": int(data_id),
            "prompt": prompt,
            "image_path": str(image_path),
            "width": int(width),
            "height": int(height),
        }
        tmp_meta = meta_path.with_name(f"{meta_path.stem}.tmp.json")
        tmp_meta.write_text(json.dumps(meta, indent=2), encoding="utf-8")
        tmp_meta.replace(meta_path)

    @staticmethod
    def load_tensors(*, paths: CachePaths, data_id: int) -> tuple[mx.array, Any, int, int]:
        tensor_path = paths.data / f"{data_id:07d}.safetensors"
        if not tensor_path.exists():
            raise FileNotFoundError(f"Cached data item not found: {tensor_path}")

        tensors, metadata = mx.load(str(tensor_path), return_metadata=True)
        schema = metadata.get("mflux_cache_schema")
        if schema is None:
            raise ValueError(
                f"Training cache missing schema metadata: {tensor_path}. Delete the cache folder and retry."
            )
        try:
            schema_version = int(schema)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Training cache schema metadata invalid ({schema}): {tensor_path}. Delete the cache folder and retry."
            ) from exc
        if schema_version != TrainingDataCache.CACHE_SCHEMA_VERSION:
            raise ValueError(
                "Training cache schema mismatch. "
                f"Expected {TrainingDataCache.CACHE_SCHEMA_VERSION}, got {schema_version}. "
                "Delete the cache folder and retry."
            )
        clean_latents = tensors["clean_latents"]
        width = int(tensors["width"].item())
        height = int(tensors["height"].item())

        if "cond" in tensors:
            cond: Any = tensors["cond"]
        else:
            cond_dict: dict[str, mx.array] = {}
            for k, v in tensors.items():
                if k.startswith("cond__"):
                    cond_dict[k[len("cond__") :]] = v
            cond = cond_dict
        return clean_latents, cond, width, height
