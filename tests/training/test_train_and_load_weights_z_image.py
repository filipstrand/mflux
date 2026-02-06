import os
import shutil

import pytest

from mflux.models.common.training.runner import TrainingRunner
from mflux.models.common.training.state.zip_util import ZipUtil
from mflux.models.z_image.variants import ZImageTurbo

CHECKPOINT = "tests/training/tmp/checkpoints/0000001_checkpoint.zip"
OUTPUT_DIR = "tests/training/tmp/checkpoints/0000001_checkpoint"
LORA_FILE = "tests/training/tmp/checkpoints/0000001_checkpoint/0000001_adapter.safetensors"


class TestTrainAndLoadWeightsZImage:
    @pytest.mark.slow
    def test_train_and_load_weights_z_image(self):
        TestTrainAndLoadWeightsZImage.delete_folder_if_exists("tests/training/tmp")

        try:
            adapter, _training_spec = TrainingRunner.train(
                config_path="tests/training/config/train.json",
                resume_path=None,
            )
            _ = adapter.model()
            del adapter, _training_spec

            ZipUtil.extract_all(zip_path=CHECKPOINT, output_dir=OUTPUT_DIR)

            loaded_model = ZImageTurbo(
                quantize=4,
                lora_paths=[LORA_FILE],
                lora_scales=[1.0],
            )
            assert loaded_model is not None
        finally:
            TestTrainAndLoadWeightsZImage.delete_folder_if_exists("tests/training/tmp")

    @staticmethod
    def delete_folder(path: str) -> None:
        return shutil.rmtree(path)

    @staticmethod
    def delete_folder_if_exists(path: str) -> None:
        if os.path.exists(path):
            shutil.rmtree(path)
