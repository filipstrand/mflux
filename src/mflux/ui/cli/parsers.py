import argparse
import json
import typing as t
from pathlib import Path

from mflux.ui import defaults as ui_defaults


# fmt: off
class CommandLineParser(argparse.ArgumentParser):

    def __init__(self, *pargs, **kwargs):
        super().__init__(*pargs, **kwargs)
        self.supports_metadata_config = False
        self.supports_image_generation = False
        self.supports_controlnet = False
        self.supports_image_to_image = False
        self.supports_lora = False

    def add_model_arguments(self, path_type: t.Literal["load", "save"] = "load", require_model_arg: bool = True) -> None:

        self.add_argument("--model", "-m", type=str, required=require_model_arg, choices=ui_defaults.MODEL_CHOICES, help=f"The model to use ({' or '.join(ui_defaults.MODEL_CHOICES)}).")

        if path_type == "load":
            self.add_argument("--path", type=str, default=None, help="Local path for loading a model from disk")
        else:
            self.add_argument("--path", type=str, required=True, help="Local path for saving a model to disk.")
        self.add_argument("--quantize",  "-q", type=int, choices=ui_defaults.QUANTIZE_CHOICES, default=None, help=f"Quantize the model ({' or '.join(map(str, ui_defaults.QUANTIZE_CHOICES))}, Default is None)")

    def add_lora_arguments(self) -> None:
        self.supports_lora = True
        self.add_argument("--lora-paths", type=str, nargs="*", default=None, help="Local safetensors for applying LORA from disk")
        self.add_argument("--lora-scales", type=float, nargs="*", default=None, help="Scaling factor to adjust the impact of LoRA weights on the model. A value of 1.0 applies the LoRA weights as they are.")

    def _add_image_generator_common_arguments(self) -> None:
        self.supports_image_generation = True
        self.add_argument("--height", type=int, default=ui_defaults.HEIGHT, help=f"Image height (Default is {ui_defaults.HEIGHT})")
        self.add_argument("--width", type=int, default=ui_defaults.WIDTH, help=f"Image width (Default is {ui_defaults.HEIGHT})")
        self.add_argument("--steps", type=int, default=None, help="Inference Steps")
        self.add_argument("--guidance", type=float, default=ui_defaults.GUIDANCE_SCALE, help=f"Guidance Scale (Default is {ui_defaults.GUIDANCE_SCALE})")

    def add_image_generator_arguments(self, supports_metadata_config=False) -> None:
        self.add_argument("--prompt", type=str, required=(not supports_metadata_config), default=None, help="The textual description of the image to generate.")
        self.add_argument("--seed", type=int, default=None, help="Entropy Seed (Default is time-based random-seed)")
        self._add_image_generator_common_arguments()
        if supports_metadata_config:
            self.add_metadata_config()

    def add_image_to_image_arguments(self, required=False) -> None:
        self.supports_image_to_image = True
        self.add_argument("--init-image-path", type=Path, required=required, default=None, help="Local path to init image")
        self.add_argument("--init-image-strength", type=float, required=False, default=ui_defaults.INIT_IMAGE_STRENGTH, help=f"Controls how strongly the init image influences the output image. A value of 0.0 means no influence. (Default is {ui_defaults.INIT_IMAGE_STRENGTH})")

    def add_batch_image_generator_arguments(self) -> None:
        self.add_argument("--prompts-file", type=Path, required=True, default=argparse.SUPPRESS, help="Local path for a file that holds a batch of prompts.")
        self.add_argument("--global-seed", type=int, default=argparse.SUPPRESS, help="Entropy Seed (used for all prompts in the batch)")
        self._add_image_generator_common_arguments()

    def add_output_arguments(self) -> None:
        self.add_argument("--metadata", action="store_true", help="Export image metadata as a JSON file.")
        self.add_argument("--output", type=str, default="image.png", help="The filename for the output image. Default is \"image.png\".")
        self.add_argument('--stepwise-image-output-dir', type=str, default=None, help='[EXPERIMENTAL] Output dir to write step-wise images and their final composite image to. This feature may change in future versions.')

    def add_controlnet_arguments(self) -> None:
        self.supports_controlnet = True
        self.add_argument("--controlnet-image-path", type=str, required=False, help="Local path of the image to use as input for controlnet.")
        self.add_argument("--controlnet-strength", type=float, default=ui_defaults.CONTROLNET_STRENGTH, help=f"Controls how strongly the control image influences the output image. A value of 0.0 means no influence. (Default is {ui_defaults.CONTROLNET_STRENGTH})")
        self.add_argument("--controlnet-save-canny", action="store_true", help="If set, save the Canny edge detection reference input image.")

    def add_metadata_config(self) -> None:
        self.supports_metadata_config = True
        self.add_argument("--config-from-metadata", "-C", type=Path, required=False, default=argparse.SUPPRESS, help="Re-use the parameters from prior metadata. Params from metadata are secondary to other args you provide.")

    def parse_args(self, **kwargs) -> argparse.Namespace:
        namespace = super().parse_args()
        if hasattr(namespace, "path") and namespace.path is not None and namespace.model is None:
            self.error("--model must be specified when using --path")

        if getattr(namespace, "config_from_metadata", False):
            prior_gen_metadata = json.load(namespace.config_from_metadata.open("rt"))

            if namespace.model is None:
                # when not provided by CLI flag, find it in the config file
                namespace.model = prior_gen_metadata.get("model", None)

            if namespace.prompt is None:
                namespace.prompt = prior_gen_metadata.get("prompt", None)

            # all configs from the metadata config defers to any explicitly defined args
            guidance_default = self.get_default("guidance")
            guidance_from_metadata = prior_gen_metadata.get("guidance")
            if namespace.guidance == guidance_default and guidance_from_metadata:
                namespace.guidance = guidance_from_metadata
            if namespace.quantize is None:
                namespace.quantize = prior_gen_metadata.get("quantize", None)
            if namespace.seed is None:
                namespace.seed = prior_gen_metadata.get("seed", None)
            if namespace.steps is None:
                namespace.steps = prior_gen_metadata.get("steps", None)

            if self.supports_lora:
                if namespace.lora_paths is None:
                    namespace.lora_paths = prior_gen_metadata.get("lora_paths", None)
                elif namespace.lora_paths:
                    # merge the loras from cli and config file
                    namespace.lora_paths = prior_gen_metadata.get("lora_paths", []) + namespace.lora_paths

                if namespace.lora_scales is None:
                    namespace.lora_scales = prior_gen_metadata.get("lora_scales", None)
                elif namespace.lora_scales:
                    # merge the loras from cli and config file
                    namespace.lora_scales = prior_gen_metadata.get("lora_scales", []) + namespace.lora_scales

            if self.supports_image_to_image:
                if namespace.init_image_path is None:
                    namespace.init_image_path = prior_gen_metadata.get("init_image_path", None)
                if namespace.init_image_strength == self.get_default("init_image_strength") and (init_img_strength_from_metadata := prior_gen_metadata.get("init_image_strength", None)):
                    namespace.init_image_strength = init_img_strength_from_metadata

            if self.supports_controlnet:
                if namespace.controlnet_image_path is None:
                    namespace.controlnet_image_path = prior_gen_metadata.get("controlnet_image_path", None)
                if namespace.controlnet_strength == self.get_default("controlnet_strength") and (cnet_strength_from_metadata := prior_gen_metadata.get("controlnet_strength", None)):
                    namespace.controlnet_strength = cnet_strength_from_metadata
                if namespace.controlnet_save_canny == self.get_default("controlnet_save_canny") and (cnet_canny_from_metadata := prior_gen_metadata.get("controlnet_save_canny", None)):
                    namespace.controlnet_save_canny = cnet_canny_from_metadata

        if namespace.model is None:
            self.error("--model / -m must be provided, or 'model' must be specified in the config file.")

        if self.supports_image_generation and namespace.prompt is None:
            # not supplied by CLI and not supplied by metadata config file
            self.error("--prompt argument required or 'prompt' required in metadata config file")

        if self.supports_image_generation and namespace.steps is None:
            namespace.steps = ui_defaults.MODEL_INFERENCE_STEPS.get(namespace.model, None)

        return namespace
