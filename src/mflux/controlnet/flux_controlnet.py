from pathlib import Path

import mlx.core as mx
from mlx import nn
from tqdm import tqdm

from mflux.config.config import ConfigControlnet
from mflux.config.model_config import ModelConfig
from mflux.config.runtime_config import RuntimeConfig
from mflux.controlnet.controlnet_util import ControlnetUtil
from mflux.controlnet.transformer_controlnet import TransformerControlnet
from mflux.error.exceptions import StopImageGenerationException
from mflux.flux.flux_initializer import FluxInitializer
from mflux.latent_creator.latent_creator import LatentCreator
from mflux.models.text_encoder.clip_encoder.clip_encoder import CLIPEncoder
from mflux.models.text_encoder.t5_encoder.t5_encoder import T5Encoder
from mflux.models.transformer.transformer import Transformer
from mflux.models.vae.vae import VAE
from mflux.post_processing.array_util import ArrayUtil
from mflux.post_processing.generated_image import GeneratedImage
from mflux.post_processing.image_util import ImageUtil
from mflux.post_processing.stepwise_handler import StepwiseHandler
from mflux.weights.model_saver import ModelSaver


class Flux1Controlnet(nn.Module):
    vae: VAE
    transformer: Transformer
    transformer_controlnet: TransformerControlnet
    t5_text_encoder: T5Encoder
    clip_text_encoder: CLIPEncoder

    def __init__(
        self,
        model_config: ModelConfig,
        quantize: int | None = None,
        local_path: str | None = None,
        lora_paths: list[str] | None = None,
        lora_scales: list[float] | None = None,
        controlnet_path: str | None = None,
    ):
        super().__init__()
        FluxInitializer.init_controlnet(
            flux_model=self,
            model_config=model_config,
            quantize=quantize,
            local_path=local_path,
            lora_paths=lora_paths,
            lora_scales=lora_scales,
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
    ) -> GeneratedImage:
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

        # 0. Encode the controlnet reference image
        controlnet_condition = ControlnetUtil.encode_image(
            vae=self.vae,
            config=config,
            controlnet_image_path=controlnet_image_path,
            controlnet_save_canny=controlnet_save_canny,
            output=output,
        )

        # 1. Create the initial latents
        latents = LatentCreator.create(
            seed=seed,
            height=config.height,
            width=config.width
        )  # fmt: off

        # 2. Embed the prompt
        t5_tokens = self.t5_tokenizer.tokenize(prompt)
        clip_tokens = self.clip_tokenizer.tokenize(prompt)
        prompt_embeds = self.t5_text_encoder(t5_tokens)
        pooled_prompt_embeds = self.clip_text_encoder(clip_tokens)

        for gen_step, t in enumerate(time_steps, 1):
            try:
                # 3.t Compute controlnet samples
                controlnet_block_samples, controlnet_single_block_samples = self.transformer_controlnet(
                    t=t,
                    config=config,
                    hidden_states=latents,
                    prompt_embeds=prompt_embeds,
                    pooled_prompt_embeds=pooled_prompt_embeds,
                    controlnet_condition=controlnet_condition,
                )

                # 4.t Predict the noise
                noise = self.transformer(
                    t=t,
                    config=config,
                    hidden_states=latents,
                    prompt_embeds=prompt_embeds,
                    pooled_prompt_embeds=pooled_prompt_embeds,
                    controlnet_block_samples=controlnet_block_samples,
                    controlnet_single_block_samples=controlnet_single_block_samples,
                )

                # 5.t Take one denoise step
                dt = config.sigmas[t + 1] - config.sigmas[t]
                latents += noise * dt

                # Handle stepwise output if enabled
                stepwise_handler.process_step(gen_step, latents)

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
