from pathlib import Path

import mlx.core as mx
from mlx import nn
from tqdm import tqdm

from mflux.config.config import Config
from mflux.config.model_config import ModelConfig
from mflux.config.runtime_config import RuntimeConfig
from mflux.error.exceptions import StopImageGenerationException
from mflux.latent_creator.latent_creator import LatentCreator
from mflux.models.text_encoder.clip_encoder.clip_encoder import CLIPEncoder
from mflux.models.text_encoder.t5_encoder.t5_encoder import T5Encoder
from mflux.models.transformer.transformer import Transformer
from mflux.models.vae.vae import VAE
from mflux.post_processing.array_util import ArrayUtil
from mflux.post_processing.generated_image import GeneratedImage
from mflux.post_processing.image_util import ImageUtil
from mflux.post_processing.stepwise_handler import StepwiseHandler
from mflux.tokenizer.clip_tokenizer import TokenizerCLIP
from mflux.tokenizer.t5_tokenizer import TokenizerT5
from mflux.tokenizer.tokenizer_handler import TokenizerHandler
from mflux.weights.model_saver import ModelSaver
from mflux.weights.weight_handler import WeightHandler
from mflux.weights.weight_handler_lora import WeightHandlerLoRA
from mflux.weights.weight_util import WeightUtil


class Flux1(nn.Module):
    def __init__(
        self,
        model_config: ModelConfig,
        quantize: int | None = None,
        local_path: str | None = None,
        lora_paths: list[str] | None = None,
        lora_scales: list[float] | None = None,
    ):
        super().__init__()
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

    def generate_image(
        self,
        seed: int,
        src_prompt: str,
        tar_prompt: str,
        src_guidance: float,
        tar_guidance: float,
        image_path: str,
        config: Config = Config(),
        stepwise_output_dir: Path = None,
    ) -> GeneratedImage:
        # Create a new runtime config based on the model type and input parameters
        config = RuntimeConfig(config, self.model_config)
        time_steps = tqdm(range(config.init_time_step, config.num_inference_steps))
        stepwise_handler = StepwiseHandler(
            flux=self,
            config=config,
            seed=seed,
            prompt=f"src_prompt: {src_prompt} | tar_prompt: {tar_prompt}",
            time_steps=time_steps,
            output_dir=stepwise_output_dir,
        )

        # 1. Create the initial latents
        image_latents = LatentCreator.encode_image(
            init_image_path=Path(image_path),
            width=config.width,
            height=config.height,
            vae=self.vae
        )  # fmt:off

        # 2a. Embed the source prompt
        t5_tokens_src = self.t5_tokenizer.tokenize(src_prompt)
        clip_tokens_src = self.clip_tokenizer.tokenize(src_prompt)
        prompt_embeds_src = self.t5_text_encoder(t5_tokens_src)
        pooled_prompt_embeds_src = self.clip_text_encoder(clip_tokens_src)
        # 2b. Embed the target prompt
        t5_tokens_tar = self.t5_tokenizer.tokenize(tar_prompt)
        clip_tokens_tar = self.clip_tokenizer.tokenize(tar_prompt)
        prompt_embeds_tar = self.t5_text_encoder(t5_tokens_tar)
        pooled_prompt_embeds_tar = self.clip_text_encoder(clip_tokens_tar)

        Z_FE = mx.array(image_latents)
        for gen_step, t in enumerate(time_steps, 1):
            try:
                if config.num_inference_steps - t > 24:
                    continue

                random_noise = mx.random.normal(shape=[1, (config.height // 16) * (config.width // 16), 64])
                Z_src = (1 - config.sigmas[t]) * image_latents + config.sigmas[t] * random_noise
                Z_tar = Z_FE + Z_src - image_latents

                # 3.t Predict the noise
                config.guidance = src_guidance
                noise_src = self.transformer.predict(
                    t=t,
                    prompt_embeds=prompt_embeds_src,
                    pooled_prompt_embeds=pooled_prompt_embeds_src,
                    hidden_states=Z_src,
                    config=config,
                )
                config.guidance = tar_guidance
                noise_tar = self.transformer.predict(
                    t=t,
                    prompt_embeds=prompt_embeds_tar,
                    pooled_prompt_embeds=pooled_prompt_embeds_tar,
                    hidden_states=Z_tar,
                    config=config,
                )

                noise_delta = noise_tar - noise_src

                # 4.t Take one denoise step
                dt = config.sigmas[t + 1] - config.sigmas[t]
                Z_FE += noise_delta * dt

                # Handle stepwise output if enabled
                stepwise_handler.process_step(gen_step, Z_FE)

                # Evaluate to enable progress tracking
                mx.eval(Z_FE)

            except KeyboardInterrupt:  # noqa: PERF203
                stepwise_handler.handle_interruption()
                raise StopImageGenerationException(f"Stopping image generation at step {t + 1}/{len(time_steps)}")

        # 5. Decode the latent array and return the image
        latents = ArrayUtil.unpack_latents(latents=Z_FE, height=config.height, width=config.width)
        decoded = self.vae.decode(latents)
        return ImageUtil.to_image(
            decoded_latents=decoded,
            seed=seed,
            prompt=f"src_prompt: {src_prompt} | tar_prompt: {tar_prompt}",
            quantization=self.bits,
            generation_time=time_steps.format_dict["elapsed"],
            lora_paths=self.lora_paths,
            lora_scales=self.lora_scales,
            init_image_path=config.init_image_path,
            init_image_strength=config.init_image_strength,
            config=config,
        )

    @staticmethod
    def from_alias(alias: str, quantize: int | None = None) -> "Flux1":
        return Flux1(
            model_config=ModelConfig.from_alias(alias),
            quantize=quantize,
        )

    def save_model(self, base_path: str) -> None:
        ModelSaver.save_model(self, self.bits, base_path)

    def freeze(self, **kwargs):
        self.vae.freeze()
        self.transformer.freeze()
        self.t5_text_encoder.freeze()
        self.clip_text_encoder.freeze()
