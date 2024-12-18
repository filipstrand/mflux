import shutil

import numpy as np

from mflux import Config, Flux1, ModelConfig
from mflux.dreambooth.dreambooth import DreamBooth
from mflux.dreambooth.dreambooth_initializer import DreamBoothInitializer
from mflux.dreambooth.state.zip_util import ZipUtil

CHECKPOINT = "tests/dreambooth/tmp/_checkpoints/0000005_checkpoint.zip"
OUTPUT_DIR = "tests/dreambooth/tmp/_checkpoints/0000005_checkpoint"
LORA_FILE = "tests/dreambooth/tmp/_checkpoints/0000005_checkpoint/0000005_adapter.safetensors"


class TestTrainAndLoadWeights:
    def test_train_and_load_weights(self):
        try:
            # Given: A small training run from scratch for 5 steps (as described in the config)...
            fluxA, runtime_config, training_spec, training_state = DreamBoothInitializer.initialize(
                config_path="tests/dreambooth/config/train.json",
                checkpoint_path=None
            )  # fmt:off
            DreamBooth.train(
                flux=fluxA,
                runtime_config=runtime_config,
                training_spec=training_spec,
                training_state=training_state
            )  # fmt: off
            # ...we generate an image with the flux instance with the trained weights
            image1 = fluxA.generate_image(
                seed=42,
                prompt="test",
                config=Config(
                    num_inference_steps=20,
                    height=128,
                    width=128,
                ),
            )
            del fluxA, runtime_config, training_spec, training_state
            # unzip so that LoRA adapter can be read later...
            ZipUtil.extract_all(zip_path=CHECKPOINT, output_dir=OUTPUT_DIR)

            # When: Loading a new Flux instance with the trained LoRA...
            fluxB = Flux1(
                model_config=ModelConfig.FLUX1_DEV,
                quantize=4,
                lora_paths=[LORA_FILE],
                lora_scales=[1.0],
            )  # fmt: off

            # ...and generating the same image from that
            image2 = fluxB.generate_image(
                seed=42,
                prompt="test",
                config=Config(
                    num_inference_steps=20,
                    height=128,
                    width=128,
                ),
            )

            # Then: We want to confirm that the images *exactly* match
            np.testing.assert_array_equal(
                np.array(image1.image),
                np.array(image2.image),
                err_msg="Generated image doesn't match reference image.",
            )
        finally:
            # cleanup
            TestTrainAndLoadWeights.delete_folder("tests/dreambooth/tmp")

    @staticmethod
    def delete_folder(path: str) -> None:
        return shutil.rmtree(path)
