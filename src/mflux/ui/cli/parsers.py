import argparse
from pathlib import Path

from mflux.ui import defaults as ui_defaults


# fmt: off
class CommandLineParser(argparse.ArgumentParser):

    def add_model_arguments(self) -> None:
        self.add_argument("--model", "-m", type=str, required=True, choices=ui_defaults.MODEL_CHOICES, help=f"The model to use ({' or '.join(ui_defaults.MODEL_CHOICES)}).")
        self.add_argument("--path", type=str, default=None, help="Local path for loading a model from disk")
        self.add_argument("--quantize",  "-q", type=int, choices=ui_defaults.QUANTIZE_CHOICES, default=None, help=f"Quantize the model ({' or '.join(map(str, ui_defaults.QUANTIZE_CHOICES))}, Default is None)")

    def add_lora_arguments(self) -> None:
        self.add_argument("--lora-paths", type=str, nargs="*", default=None, help="Local safetensors for applying LORA from disk")
        self.add_argument("--lora-scales", type=float, nargs="*", default=None, help="Scaling factor to adjust the impact of LoRA weights on the model. A value of 1.0 applies the LoRA weights as they are.")

    def _add_image_generator_common_arguments(self) -> None:
        self.add_argument("--height", type=int, default=ui_defaults.HEIGHT, help=f"Image height (Default is {ui_defaults.HEIGHT})")
        self.add_argument("--width", type=int, default=ui_defaults.WIDTH, help=f"Image width (Default is {ui_defaults.HEIGHT})")
        self.add_argument("--steps", type=int, default=None, help="Inference Steps")
        self.add_argument("--guidance", type=float, default=ui_defaults.GUIDANCE_SCALE, help=f"Guidance Scale (Default is {ui_defaults.GUIDANCE_SCALE})")

    def add_image_generator_arguments(self) -> None:
        self.add_argument("--prompt", type=str, required=True, help="The textual description of the image to generate.")
        self.add_argument("--seed", type=int, default=None, help="Entropy Seed (Default is time-based random-seed)")
        self._add_image_generator_common_arguments()

    def add_batch_image_generator_arguments(self) -> None:
        self.add_argument("--prompts-file", type=Path, required=True, help="Local path for a file that holds a batch of prompts.")
        self.add_argument("--global-seed", type=int, default=None, help="Entropy Seed (used for all prompts in the batch)")
        self._add_image_generator_common_arguments()

    def add_output_arguments(self) -> None:
        self.add_argument("--metadata", action="store_true", help="Export image metadata as a JSON file.")
        self.add_argument("--output", type=str, default="image.png", help="The filename for the output image. Default is \"image.png\".")
        self.add_argument('--stepwise-image-output-dir', type=str, default=None, help='[EXPERIMENTAL] Output dir to write step-wise images and their final composite image to. This feature may change in future versions.')

    def add_controlnet_arguments(self) -> None:
        self.add_argument("--controlnet-image-path", type=str, required=True, help="Local path of the image to use as input for controlnet.")
        self.add_argument("--controlnet-strength", type=float, default=ui_defaults.CONTROLNET_STRENGTH, help=f"Controls how strongly the control image influences the output image. A value of 0.0 means no influence. (Default is {ui_defaults.CONTROLNET_STRENGTH})")
        self.add_argument("--controlnet-save-canny", action="store_true", help="If set, save the Canny edge detection reference input image.")

    def parse_args(self, **kwargs) -> argparse.Namespace:
        namespace = super().parse_args()
        if hasattr(namespace, "path") and namespace.path is not None and namespace.model is None:
            namespace.error("--model must be specified when using --path")
        if hasattr(namespace, "steps") and namespace.steps is None:
            namespace.steps = ui_defaults.MODEL_INFERENCE_STEPS.get(namespace.model, None)
        return namespace
