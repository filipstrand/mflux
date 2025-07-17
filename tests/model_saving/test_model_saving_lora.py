import os
import shutil
from pathlib import Path

import numpy as np

from mflux.config.config import Config
from mflux.config.model_config import ModelConfig
from mflux.flux.flux import Flux1

PATH = "tests/4bit/"


class TestModelSavingLora:
    def test_save_and_load_4bit_model_with_lora(self):
        # Clean up any existing temporary directories from previous test runs
        TestModelSavingLora.delete_folder_if_exists(PATH)

        try:
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
                lora_paths=TestModelSavingLora.get_lora_path(),
                lora_scales=[1.0],
            )
            image1 = fluxB.generate_image(
                seed=44,
                prompt="mkym this is made of wool, pizza",
                config=Config(
                    num_inference_steps=2,
                    height=341,
                    width=768,
                ),
            )
            del fluxB

            # when loading the quantized model from a local path (also without specifying bits) with a LoRA...
            fluxC = Flux1(
                model_config=ModelConfig.schnell(),
                local_path=PATH,
                lora_paths=TestModelSavingLora.get_lora_path(),
                lora_scales=[1.0],
            )

            # ...and generating the identical image
            image2 = fluxC.generate_image(
                seed=44,
                prompt="mkym this is made of wool, pizza",
                config=Config(
                    num_inference_steps=2,
                    height=341,
                    width=768,
                ),
            )

            # then we confirm that we get the exact *identical* image in both cases
            np.testing.assert_array_equal(
                np.array(image1.image),
                np.array(image2.image),
                err_msg="image2 doesn't match image1.",
            )

        finally:
            # cleanup
            TestModelSavingLora.delete_folder(PATH)

    @staticmethod
    def delete_folder(path: str) -> None:
        return shutil.rmtree(path)

    @staticmethod
    def delete_folder_if_exists(path: str) -> None:
        if os.path.exists(path):
            shutil.rmtree(path)
            print(f"Deleted folder: {path}")
        else:
            print(f"Folder does not exist: {path}")

    @staticmethod
    def resolve_path(path) -> Path | None:
        if path is None:
            return None
        return Path(__file__).parent.parent / "resources" / path

    @staticmethod
    def get_lora_path() -> list[str]:
        path = TestModelSavingLora.resolve_path("FLUX-dev-lora-MiaoKa-Yarn-World.safetensors")
        return [str(path)]
