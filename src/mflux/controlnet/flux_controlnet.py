import logging
from typing import TYPE_CHECKING

import mlx.core as mx
from mlx import nn
from tqdm import tqdm

from mflux.config.config import ConfigControlnet
from mflux.config.model_config import ModelConfig
from mflux.config.runtime_config import RuntimeConfig
from mflux.controlnet.controlnet_util import ControlnetUtil
from mflux.controlnet.transformer_controlnet import TransformerControlnet
from mflux.controlnet.weight_handler_controlnet import WeightHandlerControlnet
from mflux.models.text_encoder.clip_encoder.clip_encoder import CLIPEncoder
from mflux.models.text_encoder.t5_encoder.t5_encoder import T5Encoder
from mflux.models.transformer.transformer import Transformer
from mflux.models.vae.vae import VAE
from mflux.post_processing.image_util import ImageUtil
from mflux.tokenizer.clip_tokenizer import TokenizerCLIP
from mflux.tokenizer.t5_tokenizer import TokenizerT5
from mflux.tokenizer.tokenizer_handler import TokenizerHandler
from mflux.weights.model_saver import ModelSaver
from mflux.weights.weight_handler import WeightHandler

if TYPE_CHECKING:
    from mflux.post_processing.generated_image import GeneratedImage


log = logging.getLogger(__name__)

CONTROLNET_ID = "InstantX/FLUX.1-dev-Controlnet-Canny"


class Flux1Controlnet:
    def __init__(
        self,
        model_config: ModelConfig,
        quantize: int | None = None,
        local_path: str | None = None,
        lora_paths: list[str] | None = None,
        lora_scales: list[float] | None = None,
        controlnet_path: str | None = None,
    ):
        self.lora_paths = lora_paths
        self.lora_scales = lora_scales
        self.model_config = model_config

        # Load and initialize the tokenizers from disk, huggingface cache, or download from huggingface
        tokenizers = TokenizerHandler(model_config.model_name, self.model_config.max_sequence_length, local_path)
        self.t5_tokenizer = TokenizerT5(tokenizers.t5, max_length=self.model_config.max_sequence_length)
        self.clip_tokenizer = TokenizerCLIP(tokenizers.clip)

        # Initialize the models
        self.vae = VAE()
        self.transformer = Transformer(model_config)
        self.t5_text_encoder = T5Encoder()
        self.clip_text_encoder = CLIPEncoder()

        # Load the weights from disk, huggingface cache, or download from huggingface
        weights = WeightHandler(
            repo_id=model_config.model_name,
            local_path=local_path,
            lora_paths=lora_paths,
            lora_scales=lora_scales
        )  # fmt: off

        # Set the loaded weights if they are not quantized
        if weights.quantization_level is None:
            self._set_model_weights(weights)

        # Optionally quantize the model here at initialization (also required if about to load quantized weights)
        self.bits = None
        if quantize is not None or weights.quantization_level is not None:
            self.bits = weights.quantization_level if weights.quantization_level is not None else quantize
            # fmt: off
            nn.quantize(self.vae, class_predicate=lambda _, m: isinstance(m, nn.Linear), group_size=64, bits=self.bits)
            nn.quantize(self.transformer, class_predicate=lambda _, m: isinstance(m, nn.Linear) and len(m.weight[1]) > 64, group_size=64, bits=self.bits)
            nn.quantize(self.t5_text_encoder, class_predicate=lambda _, m: isinstance(m, nn.Linear), group_size=64, bits=self.bits)
            nn.quantize(self.clip_text_encoder, class_predicate=lambda _, m: isinstance(m, nn.Linear), group_size=64, bits=self.bits)
            # fmt: on

        # If loading previously saved quantized weights, the weights must be set after modules have been quantized
        if weights.quantization_level is not None:
            self._set_model_weights(weights)

        weights_controlnet, controlnet_quantization_level, controlnet_config = WeightHandlerControlnet.load_controlnet_transformer(
            controlnet_id=CONTROLNET_ID
        )  # fmt: off
        self.transformer_controlnet = TransformerControlnet(
            model_config=model_config,
            num_blocks=controlnet_config["num_layers"],
            num_single_blocks=controlnet_config["num_single_layers"],
        )

        if controlnet_quantization_level is None:
            self.transformer_controlnet.update(weights_controlnet)

        self.bits = None
        if quantize is not None or controlnet_quantization_level is not None:
            self.bits = controlnet_quantization_level if controlnet_quantization_level is not None else quantize
            nn.quantize(self.transformer_controlnet, class_predicate=lambda _, m: isinstance(m, nn.Linear) and len(m.weight[1]) > 128, group_size=128, bits=self.bits)  # fmt: off

        if controlnet_quantization_level is not None:
            self.transformer_controlnet.update(weights_controlnet)

    def generate_image(
            self,
            seed: int,
            prompt: str,
            output: str,
            controlnet_image_path: str,
            controlnet_save_canny: bool = False,
            config: ConfigControlnet = ConfigControlnet()
    ) -> "GeneratedImage":  # fmt: off
        # Create a new runtime config based on the model type and input parameters
        config = RuntimeConfig(config, self.model_config)
        time_steps = tqdm(range(config.num_inference_steps))

        # Embedd the controlnet reference image
        control_image = ImageUtil.load_image(controlnet_image_path)
        control_image = ControlnetUtil.scale_image(config.height, config.width, control_image)
        control_image = ControlnetUtil.preprocess_canny(control_image)
        if controlnet_save_canny:
            ControlnetUtil.save_canny_image(control_image, output)
        controlnet_cond = ImageUtil.to_array(control_image)
        controlnet_cond = self.vae.encode(controlnet_cond)
        controlnet_cond = (controlnet_cond / self.vae.scaling_factor) + self.vae.shift_factor
        controlnet_cond = Flux1Controlnet._pack_latents(controlnet_cond, config.height, config.width)

        # 1. Create the initial latents
        latents = mx.random.normal(
            shape=[1, (config.height // 16) * (config.width // 16), 64],
            key=mx.random.key(seed)
        )  # fmt: off

        # 2. Embedd the prompt
        t5_tokens = self.t5_tokenizer.tokenize(prompt)
        clip_tokens = self.clip_tokenizer.tokenize(prompt)
        prompt_embeds = self.t5_text_encoder.forward(t5_tokens)
        pooled_prompt_embeds = self.clip_text_encoder.forward(clip_tokens)

        for t in time_steps:
            # Compute controlnet samples
            controlnet_block_samples, controlnet_single_block_samples = self.transformer_controlnet.forward(
                t=t,
                prompt_embeds=prompt_embeds,
                pooled_prompt_embeds=pooled_prompt_embeds,
                hidden_states=latents,
                controlnet_cond=controlnet_cond,
                config=config,
            )

            # 3.t Predict the noise
            noise = self.transformer.predict(
                t=t,
                prompt_embeds=prompt_embeds,
                pooled_prompt_embeds=pooled_prompt_embeds,
                hidden_states=latents,
                config=config,
                controlnet_block_samples=controlnet_block_samples,
                controlnet_single_block_samples=controlnet_single_block_samples,
            )

            # 4.t Take one denoise step
            dt = config.sigmas[t + 1] - config.sigmas[t]
            latents += noise * dt

            # Evaluate to enable progress tracking
            mx.eval(latents)

        # 5. Decode the latent array and return the image
        latents = Flux1Controlnet._unpack_latents(latents, config.height, config.width)
        decoded = self.vae.decode(latents)
        return ImageUtil.to_image(
            decoded_latents=decoded,
            seed=seed,
            prompt=prompt,
            quantization=self.bits,
            generation_time=time_steps.format_dict["elapsed"],
            lora_paths=self.lora_paths,
            lora_scales=self.lora_scales,
            config=config,
            controlnet_image_path=controlnet_image_path,
        )

    @staticmethod
    def _unpack_latents(latents: mx.array, height: int, width: int) -> mx.array:
        latents = mx.reshape(latents, (1, height // 16, width // 16, 16, 2, 2))
        latents = mx.transpose(latents, (0, 3, 1, 4, 2, 5))
        latents = mx.reshape(latents, (1, 16, height // 16 * 2, width // 16 * 2))
        return latents

    @staticmethod
    def _pack_latents(latents: mx.array, height: int, width: int) -> mx.array:
        latents = mx.reshape(latents, (1, 16, height // 16, 2, width // 16, 2))
        latents = mx.transpose(latents, (0, 2, 4, 1, 3, 5))
        latents = mx.reshape(latents, (1, (width // 16) * (height // 16), 64))
        return latents

    def _set_model_weights(self, weights):
        self.vae.update(weights.vae)
        self.transformer.update(weights.transformer)
        self.t5_text_encoder.update(weights.t5_encoder)
        self.clip_text_encoder.update(weights.clip_encoder)

    def save_model(self, base_path: str) -> None:
        ModelSaver.save_model(self, self.bits, base_path)
        ModelSaver.save_weights(base_path, self.bits, self.transformer_controlnet, "transformer_controlnet")
