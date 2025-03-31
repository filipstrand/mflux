import os
import shutil

import mlx.core as mx

from mflux.dreambooth.dreambooth import DreamBooth
from mflux.dreambooth.dreambooth_initializer import DreamBoothInitializer
from mflux.dreambooth.state.zip_util import ZipUtil

CHECKPOINT_3 = "tests/dreambooth/tmp/_checkpoints/0000003_checkpoint.zip"
CHECKPOINT_4 = "tests/dreambooth/tmp/_checkpoints/0000004_checkpoint.zip"
CHECKPOINT_5 = "tests/dreambooth/tmp/_checkpoints/0000005_checkpoint.zip"


class TestResumeTraining:
    def test_resume_training(self):
        # Clean up any existing temporary directories from previous test runs
        TestResumeTraining.delete_folder_if_exists("tests/dreambooth/tmp")

        try:
            # Given: A small training run from scratch for 5 steps (as described in the config)...
            fluxA, runtime_config, training_spec, training_state = DreamBoothInitializer.initialize(
                config_path="tests/dreambooth/config/train.json",
                checkpoint_path=None,
            )
            DreamBooth.train(
                flux=fluxA,
                runtime_config=runtime_config,
                training_spec=training_spec,
                training_state=training_state,
            )
            del fluxA, runtime_config, training_spec, training_state
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
            fluxB, runtime_config, training_spec, training_state = DreamBoothInitializer.initialize(
                config_path=None,
                checkpoint_path=CHECKPOINT_3,
            )
            DreamBooth.train(
                flux=fluxB,
                runtime_config=runtime_config,
                training_spec=training_spec,
                training_state=training_state,
            )
            del fluxB, runtime_config, training_spec, training_state
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
