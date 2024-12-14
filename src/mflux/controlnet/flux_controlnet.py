import logging
from pathlib import Path
from typing import TYPE_CHECKING

import mlx.core as mx
from tqdm import tqdm

from mflux.config.config import ConfigControlnet
from mflux.config.model_config import ModelConfig
from mflux.config.runtime_config import RuntimeConfig
from mflux.controlnet.controlnet_util import ControlnetUtil
from mflux.controlnet.transformer_controlnet import TransformerControlnet
from mflux.controlnet.weight_handler_controlnet import WeightHandlerControlnet
from mflux.error.exceptions import StopImageGenerationException
from mflux.latent_creator.latent_creator import LatentCreator
from mflux.models.text_encoder.clip_encoder.clip_encoder import CLIPEncoder
from mflux.models.text_encoder.t5_encoder.t5_encoder import T5Encoder
from mflux.models.transformer.transformer import Transformer
from mflux.models.vae.vae import VAE
from mflux.post_processing.array_util import ArrayUtil
from mflux.post_processing.image_util import ImageUtil
from mflux.post_processing.stepwise_handler import StepwiseHandler
from mflux.tokenizer.clip_tokenizer import TokenizerCLIP
from mflux.tokenizer.t5_tokenizer import TokenizerT5
from mflux.tokenizer.tokenizer_handler import TokenizerHandler
from mflux.weights.model_saver import ModelSaver
from mflux.weights.weight_handler import WeightHandler
from mflux.weights.weight_handler_lora import WeightHandlerLoRA
from mflux.weights.weight_util import WeightUtil

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

        # Set the weights and quantize the model
        weights = WeightHandler.load_regular_weights(repo_id=model_config.model_name, local_path=local_path)
        self.bits = WeightUtil.set_weights_and_quantize(
            quantize_arg=quantize,
            weights=weights,
            vae=self.vae,
            transformer=self.transformer,
            t5_text_encoder=self.t5_text_encoder,
            clip_text_encoder=self.clip_text_encoder,
        )

        # Set LoRA weights
        lora_weights = WeightHandlerLoRA.load_lora_weights(transformer=self.transformer, lora_files=lora_paths, lora_scales=lora_scales)  # fmt:off
        WeightHandlerLoRA.set_lora_weights(transformer=self.transformer, loras=lora_weights)

        # Set Controlnet weights
        weights_controlnet = WeightHandlerControlnet.load_controlnet_transformer(controlnet_id=CONTROLNET_ID)
        self.transformer_controlnet = TransformerControlnet(model_config=model_config, num_blocks=weights_controlnet.config["num_layers"], num_single_blocks=weights_controlnet.config["num_single_layers"])  # fmt:off
        WeightUtil.set_controlnet_weights_and_quantize(
            quantize_arg=quantize,
            weights=weights_controlnet,
            transformer_controlnet=self.transformer_controlnet,
        )

    def generate_image(
            self,
            seed: int,
            prompt: str,
            output: str,
            controlnet_image_path: str,
            controlnet_save_canny: bool = False,
            config: ConfigControlnet = ConfigControlnet(),
            stepwise_output_dir: Path = None,
    ) -> "GeneratedImage":  # fmt: off
        # Create a new runtime config based on the model type and input parameters
        config = RuntimeConfig(config, self.model_config)
        time_steps = tqdm(range(config.num_inference_steps))
        stepwise_handler = StepwiseHandler(
            flux=self,
            config=config,
            seed=seed,
            prompt=prompt,
            time_steps=time_steps,
            output_dir=stepwise_output_dir,
        )

        # Embed the controlnet reference image
        control_image = ImageUtil.load_image(controlnet_image_path)
        control_image = ControlnetUtil.scale_image(config.height, config.width, control_image)
        control_image = ControlnetUtil.preprocess_canny(control_image)
        if controlnet_save_canny:
            ControlnetUtil.save_canny_image(control_image, output)
        controlnet_cond = ImageUtil.to_array(control_image)
        controlnet_cond = self.vae.encode(controlnet_cond)
        controlnet_cond = (controlnet_cond / self.vae.scaling_factor) + self.vae.shift_factor
        controlnet_cond = ArrayUtil.pack_latents(latents=controlnet_cond, height=config.height, width=config.width)

        # 1. Create the initial latents
        latents = LatentCreator.create(seed=seed, height=config.height, width=config.width)

        # 2. Embed the prompt
        t5_tokens = self.t5_tokenizer.tokenize(prompt)
        clip_tokens = self.clip_tokenizer.tokenize(prompt)
        prompt_embeds = self.t5_text_encoder(t5_tokens)
        pooled_prompt_embeds = self.clip_text_encoder(clip_tokens)

        for t in time_steps:
            try:
                # Compute controlnet samples
                controlnet_block_samples, controlnet_single_block_samples = self.transformer_controlnet(
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

                # Handle stepwise output if enabled
                stepwise_handler.process_step(t, latents)

                # Evaluate to enable progress tracking
                mx.eval(latents)

            except KeyboardInterrupt:  # noqa: PERF203
                stepwise_handler.handle_interruption()
                raise StopImageGenerationException(f"Stopping image generation at step {t + 1}/{len(time_steps)}")

        # 5. Decode the latent array and return the image
        latents = ArrayUtil.unpack_latents(latents=latents, height=config.height, width=config.width)
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

    def save_model(self, base_path: str) -> None:
        ModelSaver.save_model(self, self.bits, base_path)
        ModelSaver.save_weights(base_path, self.bits, self.transformer_controlnet, "transformer_controlnet")
