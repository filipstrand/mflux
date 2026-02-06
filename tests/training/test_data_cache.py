from __future__ import annotations

from pathlib import Path

import mlx.core as mx
import pytest

from mflux.models.common.training.dataset.data_cache import TrainingDataCache
from mflux.models.common.training.dataset.disk_backed_data import DiskBackedData
from mflux.models.common.training.state.training_spec import DataSpec


@pytest.mark.fast
def test_data_cache_roundtrip_flux_style_cond(tmp_path: Path):
    data_root = tmp_path / "data"
    data_root.mkdir(parents=True, exist_ok=True)
    paths = TrainingDataCache.wipe_and_init(data_root=data_root)

    clean_latents = mx.ones((1, 2, 3)).astype(mx.bfloat16)
    cond = {
        "prompt_embeds": mx.zeros((1, 4, 5)).astype(mx.bfloat16),
        "pooled_prompt_embeds": mx.ones((1, 6)).astype(mx.bfloat16),
    }

    TrainingDataCache.save_item(
        paths=paths,
        data_id=0,
        prompt="p0",
        image_path=data_root / "img0.png",
        width=512,
        height=768,
        clean_latents=clean_latents,
        cond=cond,
    )

    loaded_latents, loaded_cond, w, h = TrainingDataCache.load_tensors(paths=paths, data_id=0)
    assert tuple(loaded_latents.shape) == (1, 2, 3)
    assert w == 512
    assert h == 768
    assert isinstance(loaded_cond, dict)
    assert set(loaded_cond.keys()) == {"prompt_embeds", "pooled_prompt_embeds"}
    assert tuple(loaded_cond["prompt_embeds"].shape) == (1, 4, 5)
    assert tuple(loaded_cond["pooled_prompt_embeds"].shape) == (1, 6)


@pytest.mark.fast
def test_disk_backed_data_loads_items(tmp_path: Path):
    data_root = tmp_path / "data"
    data_root.mkdir(parents=True, exist_ok=True)
    paths = TrainingDataCache.wipe_and_init(data_root=data_root)

    TrainingDataCache.save_item(
        paths=paths,
        data_id=0,
        prompt="hello",
        image_path=data_root / "img0.png",
        width=256,
        height=256,
        clean_latents=mx.zeros((1, 7, 8)),
        cond=mx.zeros((10, 2560)),
    )

    specs = [DataSpec(image=data_root / "img0.png", input_image=None, prompt="hello")]
    seq = DiskBackedData(data_specs=specs, cache_paths=paths)
    ex = seq[0]
    assert ex.data_id == 0
    assert ex.prompt == "hello"
    assert ex.width == 256
    assert ex.height == 256
    assert tuple(ex.clean_latents.shape) == (1, 7, 8)
    assert tuple(ex.cond.shape) == (10, 2560)
