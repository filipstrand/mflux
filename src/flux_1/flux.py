from mimetypes import init
import re
import PIL
import mlx.core as mx
from tqdm import tqdm

from flux_1.config.config import Config, ConfigImg2Img
from flux_1.config.runtime_config import RuntimeConfig
from flux_1.latent_creator.latent_creator import LatentCreator
from flux_1.config.model_config import ModelConfig
from flux_1.models.text_encoder.clip_encoder.clip_encoder import CLIPEncoder
from flux_1.models.text_encoder.t5_encoder.t5_encoder import T5Encoder
from flux_1.models.transformer.transformer import Transformer
from flux_1.models.vae.vae import VAE
from flux_1.post_processing.image_util import ImageUtil
from flux_1.tokenizer.clip_tokenizer import TokenizerCLIP
from flux_1.tokenizer.t5_tokenizer import TokenizerT5
from flux_1.tokenizer.tokenizer_handler import TokenizerHandler
from flux_1.weights.weight_handler import WeightHandler

import logging
log = logging.getLogger(__name__)


class Flux1:

    def __init__(self, repo_id: str):
        self.model_config = ModelConfig.from_repo(repo_id)

        # Initialize the tokenizers
        tokenizers = TokenizerHandler.load_from_disk_or_huggingface(repo_id, self.model_config.max_sequence_length)
        self.t5_tokenizer = TokenizerT5(tokenizers.t5, max_length=self.model_config.max_sequence_length)
        self.clip_tokenizer = TokenizerCLIP(tokenizers.clip)

        # Initialize the models
        weights = WeightHandler.load_from_disk_or_huggingface(repo_id)
        self.vae = VAE(weights.vae)
        self.transformer = Transformer(weights.transformer)
        self.t5_text_encoder = T5Encoder(weights.t5_encoder)
        self.clip_text_encoder = CLIPEncoder(weights.clip_encoder)

    @staticmethod
    def from_repo(repo_id: str) -> "Flux1":
        return Flux1(repo_id)

    @staticmethod
    def from_alias(alias: str) -> "Flux1":
        return Flux1(ModelConfig.from_alias(alias).model_name)

    def generate_image(self, seed: int, prompt: str, config: Config = Config()) -> PIL.Image.Image:
        # Create a new runtime config based on the model type and input parameters
        runtime_config = RuntimeConfig(config, self.model_config)

        # Create the latents
        latents = LatentCreator.create(runtime_config.height, runtime_config.width, seed)

        return self._generate_from_latents(latents, prompt, runtime_config)

    def _generate_from_latents(self, latents: mx.array, prompt: str, config: RuntimeConfig) -> PIL.Image.Image:
        # Embed the prompt
        t5_tokens = self.t5_tokenizer.tokenize(prompt)
        clip_tokens = self.clip_tokenizer.tokenize(prompt)
        prompt_embeds = self.t5_text_encoder.forward(t5_tokens)
        pooled_prompt_embeds = self.clip_text_encoder.forward(clip_tokens)

        for t in tqdm(config.inference_steps, desc="Generating image", unit="step"):
            noise = self.transformer.predict(
                t=t,
                prompt_embeds=prompt_embeds,
                pooled_prompt_embeds=pooled_prompt_embeds,
                hidden_states=latents,
                config=config,
            )

            # Take one denoise step
            dt = config.sigmas[t + 1] - config.sigmas[t]
            latents += noise * dt

            # To enable progress tracking
            mx.eval(latents)

        # Decode the latent array
        latents = Flux1._unpack_latents(latents, config.height, config.width)
        decoded = self.vae.decode(latents)
        return ImageUtil.to_image(decoded)
    
    @staticmethod
    def _pack_latents(latents: mx.array, height: int, width: int) -> mx.array:
        latents = mx.reshape(latents, (1, 16, height // 16, 2, width // 16, 2))
        latents = mx.transpose(latents, (0, 2, 4, 1, 3, 5))
        latents = mx.reshape(latents, (1, (width // 16) * (height // 16), 64))
        return latents

    @staticmethod
    def _unpack_latents(latents: mx.array, height: int, width: int) -> mx.array:
        latents = mx.reshape(latents, (1, height // 16, width // 16, 16, 2, 2))
        latents = mx.transpose(latents, (0, 3, 1, 4, 2, 5))
        latents = mx.reshape(latents, (1, 16, height // 16 * 2, width // 16 * 2))
        return latents
    

class Flux1Img2Img(Flux1):

    def __init__(self, repo_id: str):
        super().__init__(repo_id)

    @staticmethod
    def from_repo(repo_id: str) -> "Flux1Img2Img":
        return Flux1Img2Img(repo_id)

    @staticmethod
    def from_alias(alias: str) -> "Flux1Img2Img":
        return Flux1Img2Img(ModelConfig.from_alias(alias).model_name)

    def generate_image(self, seed: int, base_image: PIL.Image.Image, prompt: str, config: ConfigImg2Img) -> PIL.Image.Image:
        # Create a new runtime config based on the model type and input parameters
        runtime_config = RuntimeConfig(config, self.model_config)

        if config.height != base_image.height or config.width != base_image.width:
            log.warning("Config height and width do not match base image. Adjusting to match base image.")
            runtime_config.height = base_image.height
            runtime_config.width = base_image.width

        noise = LatentCreator.create(runtime_config.height, runtime_config.width, seed)
        base_image = ImageUtil.to_array(base_image)
        image_latents = self.vae.encode(base_image)
        image_latents = self._pack_latents(image_latents, runtime_config.height, runtime_config.width)
        latents = runtime_config.sigmas[config.init_timestep] * noise + (1.0 - runtime_config.sigmas[config.init_timestep]) * image_latents

        return self._generate_from_latents(latents, prompt, runtime_config)

