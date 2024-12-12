import mlx
import numpy as np

from mflux import Config, Flux1, ModelConfig
from mflux.dreambooth.dreambooth import DreamBooth
from mflux.dreambooth.dreambooth_initializer import DreamBoothInitializer
from mflux.dreambooth.lora_layers.lora_layers import LoRALayers
from mflux.dreambooth.state.zip_util import ZipUtil
from tests.dreambooth.test_resume_training import TestResumeTraining

CHECKPOINT = "tests/dreambooth/tmp/_checkpoints/0000005_checkpoint.zip"
OUTPUT_DIR = "tests/dreambooth/tmp/_checkpoints/0000005_checkpoint"
LORA_FILE = "tests/dreambooth/tmp/_checkpoints/0000005_checkpoint/0000005_adapter.safetensors"


class TestTrainAndLoadWeights:
    def test_train_and_load_weights(self):
        try:
            # Given: A small training run from scratch for 5 steps (as described in the config)...
            flux, runtime_config, training_spec, training_state = DreamBoothInitializer.initialize(
                config_path="tests/dreambooth/config/train.json",
                checkpoint_path=None
            )  # fmt:off
            DreamBooth.train(
                flux=flux,
                runtime_config=runtime_config,
                training_spec=training_spec,
                training_state=training_state
            )  # fmt: off
            # ...we generate an image with the flux instance with the trained weights
            image1 = flux.generate_image(
                seed=42,
                prompt="test",
                config=Config(
                    num_inference_steps=20,
                    height=128,
                    width=128,
                ),
            )
            del flux, runtime_config, training_spec, training_state
            # unzip so that LoRA adapter can be read later...
            ZipUtil.extract_all(zip_path=CHECKPOINT, output_dir=OUTPUT_DIR)

            # When: Loading a new Flux instance with the trained LoRA...
            flux = Flux1(
                model_config=ModelConfig.FLUX1_DEV,
                quantize=4,
            )  # fmt: off
            lora = LoRALayers.from_transformer_template(mlx.core.load(str(LORA_FILE)), flux.transformer)
            flux.set_lora_layers(lora)

            # ...and generating the same image from that
            image2 = flux.generate_image(
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
            TestResumeTraining.delete_folder("tests/dreambooth/tmp")

    def test_1111(self):
        flux = Flux1(
            model_config=ModelConfig.FLUX1_DEV,
            quantize=4,
            lora_paths=["/Users/filipstrand/Desktop/0000005_adapter.safetensors"]
            # lora_paths=["/Users/filipstrand/Desktop/a.safetensors"]
            # lora_paths=["/Users/filipstrand/Desktop/b.safetensors"]
        )  # fmt: off

        print("hej")

    def test_2222(self):
        flux = Flux1(
            model_config=ModelConfig.FLUX1_DEV,
            quantize=4,
        )  # fmt: off
        lora = LoRALayers.from_transformer_template(
            mlx.core.load(str("/Users/filipstrand/Desktop/0000005_adapter.safetensors")), flux
        )
        # lora = LoRALayers.from_file(mlx.core.load(str("/Users/filipstrand/Desktop/a.safetensors")), flux)
        # lora = LoRALayers.from_file(mlx.core.load(str("/Users/filipstrand/Desktop/b.safetensors")), flux)
        flux.set_lora_layers(lora)

        print("hej")
