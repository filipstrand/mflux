import mlx.core as mx
from mlx import nn
from tqdm import tqdm

from mflux.callbacks.callbacks import Callbacks
from mflux.config.config import Config
from mflux.config.model_config import ModelConfig
from mflux.config.runtime_config import RuntimeConfig
from mflux.models.common.latent_creator.latent_creator import Img2Img, LatentCreator
from mflux.models.common.weights.model_saver import ModelSaver
from mflux.models.fibo.fibo_initializer import FIBOInitializer
from mflux.models.fibo.latent_creator.fibo_latent_creator import FiboLatentCreator
from mflux.models.fibo.model.fibo_text_encoder.prompt_encoder import PromptEncoder
from mflux.models.fibo.model.fibo_text_encoder.smol_lm3_3b_text_encoder import SmolLM3_3B_TextEncoder
from mflux.models.fibo.model.fibo_transformer import FiboTransformer
from mflux.models.fibo.model.fibo_vae.wan_2_2_vae import Wan2_2_VAE
from mflux.models.fibo.tokenizer.fibo_tokenizer import TokenizerFibo
from mflux.utils.exceptions import StopImageGenerationException
from mflux.utils.generated_image import GeneratedImage
from mflux.utils.image_util import ImageUtil


class FIBO(nn.Module):
    vae: Wan2_2_VAE
    transformer: FiboTransformer
    text_encoder: SmolLM3_3B_TextEncoder
    fibo_tokenizer: TokenizerFibo

    def __init__(
        self,
        model_config: ModelConfig,
        quantize: int | None = None,
        local_path: str | None = None,
    ):
        super().__init__()
        self.model_config = model_config
        self.bits = quantize
        self.local_path = local_path

        FIBOInitializer.init(
            fibo_model=self,
            model_config=model_config,
            quantize=quantize,
            local_path=local_path,
        )

    def generate_image(
        self,
        seed: int,
        prompt: str,
        config: Config,
        negative_prompt: str | None = None,
    ) -> GeneratedImage:
        # 0. Create a new runtime config based on the model type and input parameters
        runtime_config = RuntimeConfig(config, self.model_config)
        time_steps = tqdm(range(runtime_config.init_time_step, runtime_config.num_inference_steps))

        # 1. Create the initial latents
        latents = LatentCreator.create_for_txt2img_or_img2img(
            seed=seed,
            height=runtime_config.height,
            width=runtime_config.width,
            img2img=Img2Img(
                vae=self.vae,
                latent_creator=FiboLatentCreator,
                image_path=runtime_config.image_path,
                sigmas=runtime_config.scheduler.sigmas,
                init_time_step=runtime_config.init_time_step,
            ),
        )

        # 2. Encode the prompt
        json_prompt, encoder_hidden_states, text_encoder_layers = PromptEncoder.encode_prompt(
            prompt=prompt,
            negative_prompt=negative_prompt,
            tokenizer=self.fibo_tokenizer,
            text_encoder=self.text_encoder,
        )

        # (Optional) Call subscribers for beginning of loop
        Callbacks.before_loop(
            seed=seed,
            prompt=json_prompt,
            latents=latents,
            config=runtime_config,
        )

        for t in time_steps:
            try:
                # 3.t Predict the noise
                noise_pred = self.transformer(
                    t=t,
                    config=runtime_config,
                    hidden_states=latents,
                    encoder_hidden_states=encoder_hidden_states,
                    text_encoder_layers=text_encoder_layers,
                )
                noise_pred = FIBO._apply_classifier_free_guidance(noise_pred, runtime_config.guidance)

                # 4.t Take one denoise step
                latents = runtime_config.scheduler.step(
                    model_output=noise_pred,
                    timestep=t,
                    sample=latents,
                )

                # (Optional) Call subscribers in-loop
                Callbacks.in_loop(
                    t=t,
                    seed=seed,
                    prompt=json_prompt,
                    latents=latents,
                    config=runtime_config,
                    time_steps=time_steps,
                )

                # (Optional) Evaluate to enable progress tracking
                mx.eval(latents)

            except KeyboardInterrupt:  # noqa: PERF203
                Callbacks.interruption(
                    t=t,
                    seed=seed,
                    prompt=json_prompt,
                    latents=latents,
                    config=runtime_config,
                    time_steps=time_steps,
                )
                raise StopImageGenerationException(
                    f"Stopping image generation at step {t + 1}/{runtime_config.num_inference_steps}"
                )

        # (Optional) Call subscribers after loop
        Callbacks.after_loop(
            seed=seed,
            prompt=json_prompt,
            latents=latents,
            config=runtime_config,
        )

        # 5. Decode the latent array and return the image
        latents = FIBO._unpack_latents(latents, runtime_config.height, runtime_config.width)
        decoded = self.vae.decode(latents)
        return ImageUtil.to_image(
            decoded_latents=decoded,
            config=runtime_config,
            seed=seed,
            prompt=json_prompt,
            quantization=self.bits,
            lora_paths=None,
            lora_scales=None,
            image_path=runtime_config.image_path,
            image_strength=runtime_config.image_strength,
            generation_time=time_steps.format_dict["elapsed"],
        )

    @staticmethod
    def _apply_classifier_free_guidance(noise_pred: mx.array, guidance: float) -> mx.array:
        half = noise_pred.shape[0] // 2
        noise_uncond = noise_pred[:half]
        noise_text = noise_pred[half:]
        return noise_uncond + guidance * (noise_text - noise_uncond)

    @staticmethod
    def _unpack_latents(latents: mx.array, height: int, width: int) -> mx.array:
        batch_size, seq_len, channels = latents.shape
        vae_scale_factor = 16
        latent_height = height // vae_scale_factor
        latent_width = width // vae_scale_factor
        latents = mx.reshape(latents, (batch_size, latent_height, latent_width, channels))
        latents = mx.transpose(latents, (0, 3, 1, 2))
        return latents

    def save_model(self, base_path: str) -> None:
        ModelSaver.save_model(
            model=self,
            bits=self.bits,
            base_path=base_path,
            tokenizers=[
                ("fibo_tokenizer.tokenizer", "tokenizer"),
            ],
            components=[
                ("vae", "vae"),
                ("transformer", "transformer"),
                ("text_encoder", "text_encoder"),
            ],
        )
