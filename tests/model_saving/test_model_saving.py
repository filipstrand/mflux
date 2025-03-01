import os
import shutil

import numpy as np

from mflux import Config, Flux1, ModelConfig

PATH = "tests/4bit/"


class TestModelSaving:
    def test_save_and_load_4bit_model(self):
        # Clean up any existing temporary directories from previous test runs
        TestModelSaving.delete_folder_if_exists(PATH)

        try:
            # given a saved quantized model (and an image from that model)
            fluxA = Flux1(
                model_config=ModelConfig.dev(),
                quantize=4,
            )
            image1 = fluxA.generate_image(
                seed=42,
                prompt="Luxury food photograph",
                config=Config(
                    num_inference_steps=15,
                    height=341,
                    width=768,
                ),
            )
            fluxA.save_model(PATH)
            del fluxA

            # when loading the quantized model (also without specifying bits)
            fluxB = Flux1(
                model_config=ModelConfig.dev(),
                local_path=PATH,
            )

            # then we can load the model and generate the identical image
            image2 = fluxB.generate_image(
                seed=42,
                prompt="Luxury food photograph",
                config=Config(
                    num_inference_steps=15,
                    height=341,
                    width=768,
                ),
            )
            np.testing.assert_array_equal(
                np.array(image1.image),
                np.array(image2.image),
                err_msg="image2 doesn't match image1.",
            )

        finally:
            # cleanup
            TestModelSaving.delete_folder(PATH)

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
