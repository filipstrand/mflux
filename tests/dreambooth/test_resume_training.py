import gc
import os
import shutil

import mlx.core as mx
import numpy as np
import pytest

from mflux.models.common.config import ModelConfig
from mflux.models.flux.variants.dreambooth.dreambooth import DreamBooth
from mflux.models.flux.variants.dreambooth.dreambooth_initializer import DreamBoothInitializer
from mflux.models.flux.variants.dreambooth.state.zip_util import ZipUtil
from mflux.models.flux.variants.txt2img.flux import Flux1

CHECKPOINT_3 = "tests/dreambooth/tmp/_checkpoints/0000003_checkpoint.zip"
CHECKPOINT_4 = "tests/dreambooth/tmp/_checkpoints/0000004_checkpoint.zip"
CHECKPOINT_5 = "tests/dreambooth/tmp/_checkpoints/0000005_checkpoint.zip"
OUTPUT_DIR = "tests/dreambooth/tmp/_checkpoints/0000005_checkpoint"
LORA_FILE = "tests/dreambooth/tmp/_checkpoints/0000005_checkpoint/0000005_adapter.safetensors"


class TestResumeTraining:
    @pytest.mark.slow
    def test_resume_training(self):
        # Clean up any existing temporary directories from previous test runs
        TestResumeTraining.delete_folder_if_exists("tests/dreambooth/tmp")

        try:
            # Given: A small training run from scratch for 5 steps (as described in the config)...
            fluxA, config, training_spec, training_state = DreamBoothInitializer.initialize(
                config_path="tests/dreambooth/config/train.json",
                checkpoint_path=None,
            )
            DreamBooth.train(
                flux=fluxA,
                config=config,
                training_spec=training_spec,
                training_state=training_state,
            )
            del fluxA, config, training_spec, training_state
            gc.collect()
            mx.clear_cache()
            # ...where we can inspect the training state after 5 runs...
            adapter_after_5_steps = ZipUtil.unzip(
                zip_path=CHECKPOINT_5,
                filename="0000005_adapter.safetensors",
                loader=lambda x: dict(mx.load(x).items()),
            )
            # ...and deleting the outputs past step 3 to be sure we regenerate them after this...
            TestResumeTraining.delete_file(CHECKPOINT_4)
            TestResumeTraining.delete_file(CHECKPOINT_5)

            # When: Resuming the training from step 3...
            fluxB, config, training_spec, training_state = DreamBoothInitializer.initialize(
                config_path=None,
                checkpoint_path=CHECKPOINT_3,
            )
            DreamBooth.train(
                flux=fluxB,
                config=config,
                training_spec=training_spec,
                training_state=training_state,
            )
            del fluxB, config, training_spec, training_state
            gc.collect()
            mx.clear_cache()
            # ...where we can inspect the training state after 2 additional runs...
            adapter_after_5_steps_resumed = ZipUtil.unzip(
                zip_path=CHECKPOINT_5,
                filename="0000005_adapter.safetensors",
                loader=lambda x: dict(mx.load(x).items()),
            )

            # Then: We want to confirm that the weights *exactly* match
            assert len(adapter_after_5_steps.keys()) == len(adapter_after_5_steps_resumed.keys())
            for key in adapter_after_5_steps.keys():
                if key in adapter_after_5_steps_resumed:
                    array1 = adapter_after_5_steps[key]
                    array2 = adapter_after_5_steps_resumed[key]
                    assert mx.array_equal(array1, array2)

            # And: We want to confirm that the resumed weights can actually be loaded into Flux for generation...
            ZipUtil.extract_all(zip_path=CHECKPOINT_5, output_dir=OUTPUT_DIR)
            flux_with_resumed_lora = Flux1(
                model_config=ModelConfig.dev(),
                quantize=4,
                lora_paths=[LORA_FILE],
                lora_scales=[1.0],
            )

            # ...and we should be able to generate with the resumed LoRA weights (in this case, the actual image is nonsense and doesn't matter)
            image = flux_with_resumed_lora.generate_image(
                seed=42,
                prompt="test",
                num_inference_steps=1,
                height=128,
                width=128,
            )

            # Basic sanity check that we got a valid image
            assert image.image is not None
            assert np.array(image.image).shape == (128, 128, 3)
        finally:
            # cleanup
            TestResumeTraining.delete_folder("tests/dreambooth/tmp")

    @staticmethod
    def delete_file(path: str) -> None:
        try:
            if os.path.exists(path):
                os.remove(path)
                print(f"Deleted ZIP file: {path}")
            else:
                print("The specified ZIP file does not exist.")
        except FileNotFoundError:
            print(f"File not found: {path}")
        except PermissionError:
            print(f"Permission denied: {path}")
        except OSError as e:
            print(f"OS error while deleting file: {e}")

    @staticmethod
    def delete_folder(path: str) -> None:
        return shutil.rmtree(path)

    @staticmethod
    def delete_folder_if_exists(path: str) -> None:
        if os.path.exists(path):
            shutil.rmtree(path)
            print(f"Deleted folder: {path}")
        else:
            print("The specified folder does not exist.")
