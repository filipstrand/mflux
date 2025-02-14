import mlx.core as mx
from mlx import nn
from tqdm import tqdm

from mflux.callbacks.callbacks import Callbacks
from mflux.config.config import Config
from mflux.config.model_config import ModelConfig
from mflux.config.runtime_config import RuntimeConfig
from mflux.flux.flux_initializer import FluxInitializer
from mflux.latent_creator.latent_creator import Img2Img, LatentCreator
from mflux.models.text_encoder.clip_encoder.clip_encoder import CLIPEncoder
from mflux.models.text_encoder.prompt_encoder import PromptEncoder
from mflux.models.text_encoder.t5_encoder.t5_encoder import T5Encoder
from mflux.models.transformer.transformer import Transformer
from mflux.models.vae.vae import VAE
from mflux.post_processing.array_util import ArrayUtil
from mflux.post_processing.generated_image import GeneratedImage
from mflux.post_processing.image_util import ImageUtil
from mflux.weights.model_saver import ModelSaver


class Flux1(nn.Module):
    vae: VAE
    transformer: Transformer
    t5_text_encoder: T5Encoder
    clip_text_encoder: CLIPEncoder

    def __init__(
        self,
        model_config: ModelConfig,
        quantize: int | None = None,
        local_path: str | None = None,
        lora_paths: list[str] | None = None,
        lora_scales: list[float] | None = None,
    ):
        super().__init__()
        FluxInitializer.init(
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
        config: Config,
    ) -> GeneratedImage:
        # 0. Create a new runtime config based on the model type and input parameters
        config = RuntimeConfig(config, self.model_config)
        time_steps = tqdm(range(config.init_time_step, config.num_inference_steps))

        # 1. Encode the reference image
        encoded_image = LatentCreator.encode_image(
            height=config.height,
            width=config.width,
            img2img=Img2Img(
                vae=self.vae,
                sigmas=None,
                init_time_step=None,
                init_image_path=config.init_image_path
            ),
        )  # fmt: off

        # 2. Create the initial latents
        config.width = 2 * config.width
        L = 128
        N = mx.random.normal(shape=[1, 16, L, L * 2], key=mx.random.key(seed))
        latents = ArrayUtil.pack_latents(latents=N, height=config.height, width=config.width)

        # 3. Encode the prompt
        prompt_embeds, pooled_prompt_embeds = PromptEncoder.encode_prompt(
            prompt=prompt,
            prompt_cache=self.prompt_cache,
            t5_tokenizer=self.t5_tokenizer,
            clip_tokenizer=self.clip_tokenizer,
            t5_text_encoder=self.t5_text_encoder,
            clip_text_encoder=self.clip_text_encoder,
        )

        # (Optional) Call subscribers for beginning of loop
        Callbacks.before_loop(
            seed=seed,
            prompt=prompt,
            latents=latents,
            config=config,
        )  # fmt: off

        for t in time_steps:
            try:
                # 4.t Predict the noise
                noise = self.transformer(
                    t=t,
                    config=config,
                    hidden_states=latents,
                    prompt_embeds=prompt_embeds,
                    pooled_prompt_embeds=pooled_prompt_embeds,
                )

                # 5.t Take one denoise step
                dt = config.sigmas[t + 1] - config.sigmas[t]
                latents_updated = latents + noise * dt

                # 6.t Unpack and make sure left side (ref image) is updated by interpolation
                unpacked = ArrayUtil.unpack_latents(latents=latents_updated, height=config.height, width=config.width)  # fmt: off
                unpacked[:, :, :, 0:L] = LatentCreator.add_noise_by_interpolation(clean=mx.array(encoded_image), noise=N[:, :, :, 0:L], sigma=config.sigmas[t])  # fmt: off
                latents = ArrayUtil.pack_latents(latents=unpacked, height=config.height, width=config.width)

                # (Optional) Call subscribes at end of loop
                Callbacks.in_loop(
                    t=t,
                    seed=seed,
                    prompt=prompt,
                    latents=latents,
                    config=config,
                    time_steps=time_steps,
                )  # fmt: off

                # (Optional) Evaluate to enable progress tracking
                mx.eval(latents)

            except KeyboardInterrupt:  # noqa: PERF203
                Callbacks.interruption(
                    t=t,
                    seed=seed,
                    prompt=prompt,
                    latents=latents,
                    config=config,
                    time_steps=time_steps,
                )

        # 7. Decode the latent array and return the image
        latents = ArrayUtil.unpack_latents(latents=latents, height=config.height, width=config.width)
        decoded = self.vae.decode(latents)
        return ImageUtil.to_image(
            decoded_latents=decoded,
            config=config,
            seed=seed,
            prompt=prompt,
            quantization=self.bits,
            lora_paths=self.lora_paths,
            lora_scales=self.lora_scales,
            init_image_path=config.init_image_path,
            init_image_strength=config.init_image_strength,
            generation_time=time_steps.format_dict["elapsed"],
        )

    @staticmethod
    def from_name(model_name: str, quantize: int | None = None) -> "Flux1":
        return Flux1(
            model_config=ModelConfig.from_name(model_name=model_name, base_model=None),
            quantize=quantize,
        )

    def save_model(self, base_path: str) -> None:
        ModelSaver.save_model(self, self.bits, base_path)

    def freeze(self, **kwargs):
        self.vae.freeze()
        self.transformer.freeze()
        self.t5_text_encoder.freeze()
        self.clip_text_encoder.freeze()
