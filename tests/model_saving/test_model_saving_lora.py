import os
import shutil
from pathlib import Path

import numpy as np
import pytest

from mflux.models.common.config import ModelConfig
from mflux.models.flux.variants.txt2img.flux import Flux1

PATH = "tests/4bit/"
SIZE_TOLERANCE_RATIO = 0.05  # allow small metadata/header differences


class TestModelSavingLora:
    LORA_FILES = [
        "FLUX-dev-lora-MiaoKa-Yarn-World.safetensors",
        "Flux_-_Renaissance_art_style.safetensors",
    ]
    LORA_SCALES = [0.6, 0.4]

    @pytest.mark.slow
    def test_save_and_load_4bit_model_with_lora(self):
        # Clean up any existing temporary directories from previous test runs
        TestModelSavingLora._delete_folder_if_exists(PATH)

        try:
            lora_paths = TestModelSavingLora._get_lora_paths()
            assert len(lora_paths) == len(TestModelSavingLora.LORA_SCALES)

            # given a saved quantized model on disk (without LoRA)...
            fluxA = Flux1(
                model_config=ModelConfig.schnell(),
                quantize=4,
            )
            fluxA.save_model(PATH)
            del fluxA

            # ...and given an 'on-the-fly' quantized model which we generate an image from
            fluxB = Flux1(
                model_config=ModelConfig.schnell(),
                quantize=4,
                lora_paths=lora_paths,
                lora_scales=TestModelSavingLora.LORA_SCALES,
            )
            image1 = fluxB.generate_image(
                seed=44,
                prompt="mkym this is made of wool, pizza",
                num_inference_steps=2,
                height=341,
                width=768,
            )
            del fluxB

            # when loading the quantized model from a local path (also without specifying bits) with a LoRA...
            fluxC = Flux1(
                model_config=ModelConfig.schnell(),
                model_path=PATH,
                lora_paths=lora_paths,
                lora_scales=TestModelSavingLora.LORA_SCALES,
            )

            # ...and generating the identical image
            image2 = fluxC.generate_image(
                seed=44,
                prompt="mkym this is made of wool, pizza",
                num_inference_steps=2,
                height=341,
                width=768,
            )

            # then we confirm that we get the exact *identical* image in both cases
            np.testing.assert_array_equal(
                np.array(image1.image),
                np.array(image2.image),
                err_msg="image2 doesn't match image1.",
            )

        finally:
            # cleanup
            TestModelSavingLora._delete_folder_if_exists(PATH)

    @pytest.mark.slow
    def test_save_with_lora_has_same_shard_count_as_base(self):
        base_path = "tests/4bit_base/"
        lora_path = "tests/4bit_with_lora/"

        TestModelSavingLora._delete_folder_if_exists(base_path)
        TestModelSavingLora._delete_folder_if_exists(lora_path)

        try:
            flux_base = Flux1(
                model_config=ModelConfig.schnell(),
                quantize=4,
            )
            flux_base.save_model(base_path)
            del flux_base

            flux_lora = Flux1(
                model_config=ModelConfig.schnell(),
                quantize=4,
                lora_paths=TestModelSavingLora._get_lora_paths(),
                lora_scales=TestModelSavingLora.LORA_SCALES,
            )
            flux_lora.save_model(lora_path)
            del flux_lora

            base_shards = list((Path(base_path) / "transformer").glob("*.safetensors"))
            lora_shards = list((Path(lora_path) / "transformer").glob("*.safetensors"))

            assert len(base_shards) == len(lora_shards), "LoRA save should not inflate transformer shard count"

            # Also assert the total saved size stays within a tight bound to catch shard bloat
            base_size = TestModelSavingLora._dir_size_bytes(base_path)
            lora_size = TestModelSavingLora._dir_size_bytes(lora_path)

            # LoRA baking should leave sizes effectively unchanged; allow a small tolerance for metadata
            max_allowed = base_size * (1 + SIZE_TOLERANCE_RATIO)
            assert lora_size <= max_allowed, f"LoRA save size grew unexpectedly: base={base_size}B vs lora={lora_size}B"
        finally:
            TestModelSavingLora._delete_folder_if_exists(base_path)
            TestModelSavingLora._delete_folder_if_exists(lora_path)

    @staticmethod
    def _delete_folder_if_exists(path: str) -> None:
        if os.path.exists(path):
            shutil.rmtree(path)
            print(f"Deleted folder: {path}")
        else:
            print(f"Folder does not exist: {path}")

    @staticmethod
    def _resolve_path(path) -> Path | None:
        if path is None:
            return None
        return Path(__file__).parent.parent / "resources" / path

    @staticmethod
    def _get_lora_paths() -> list[str]:
        resolved_paths = [TestModelSavingLora._resolve_path(path) for path in TestModelSavingLora.LORA_FILES]
        missing = [p for p in resolved_paths if p is None or not p.exists()]
        if missing:
            missing_names = ", ".join(sorted(p.name if p else "unknown" for p in missing))
            pytest.skip(f"Missing local LoRA test asset(s): {missing_names}")
        return [str(path) for path in resolved_paths if path is not None]

    @staticmethod
    def _dir_size_bytes(path: str | Path) -> int:
        p = Path(path)
        if not p.exists():
            return 0
        return sum(f.stat().st_size for f in p.rglob("*") if f.is_file())
