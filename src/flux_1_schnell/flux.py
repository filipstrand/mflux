from re import L
import re
import PIL
from cv2 import log
import mlx.core as mx
from PIL import Image
from sympy import im
from tqdm import tqdm

from flux_1_schnell.config.config import Config
from flux_1_schnell.latent_creator.latent_creator import LatentCreator
from flux_1_schnell.models.text_encoder.clip_encoder.clip_encoder import CLIPEncoder
from flux_1_schnell.models.text_encoder.t5_encoder.t5_encoder import T5Encoder
from flux_1_schnell.models.transformer.transformer import Transformer
from flux_1_schnell.models.vae.vae import VAE
from flux_1_schnell.post_processing.image_util import ImageUtil
from flux_1_schnell.tokenizer.clip_tokenizer import TokenizerCLIP
from flux_1_schnell.tokenizer.t5_tokenizer import TokenizerT5
from flux_1_schnell.tokenizer.tokenizer_handler import TokenizerHandler
from flux_1_schnell.weights.weight_handler import WeightHandler
import logging

log = logging.getLogger(__name__)


class Flux1:

    def __init__(self, repo_id: str, max_sequence_length: int = 512):
        self.is_dev = "FLUX.1-dev" in repo_id
        tokenizers = TokenizerHandler.load_from_disk_or_huggingface(repo_id, max_sequence_length)
        self.t5_tokenizer = TokenizerT5(tokenizers.t5, max_length=max_sequence_length)
        self.clip_tokenizer = TokenizerCLIP(tokenizers.clip)

        weights = WeightHandler.load_from_disk_or_huggingface(repo_id)
        self.vae = VAE(weights.vae)
        self.transformer = Transformer(weights.transformer)
        self.t5_text_encoder = T5Encoder(weights.t5_encoder)
        self.clip_text_encoder = CLIPEncoder(weights.clip_encoder)

    def _generate_from_latents(self, latents: mx.array, prompt: str, config: Config) -> PIL.Image.Image:
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
                config=config
            )

            dt = config.sigmas[t + 1] - config.sigmas[t]
            latents += noise * dt

            mx.eval(latents)

        latents = Flux1._unpack_latents(latents, config.height, config.width)
        decoded = self.vae.decode(latents)
        return ImageUtil.to_image(decoded)

    def generate_image(self, seed: int, prompt: str, config: Config = Config()) -> PIL.Image.Image:
        if self.is_dev:
            config.shift_sigmas()
        latents = LatentCreator.create(config.height, config.width, seed)

        return self._generate_from_latents(latents, prompt, config)
    
    @staticmethod
    def _pack_latents(latents: mx.array, height: int, width: int) -> mx.array:
        latents = mx.reshape(latents, (1, 16, height // 16, 2, width // 16, 2))
        latents = mx.transpose(latents, (0, 2, 4, 1, 3, 5))
        latents = mx.reshape(latents, (1, (height // 16) * (width // 16), 64))
        return latents

    @staticmethod
    def _unpack_latents(latents: mx.array, height: int, width: int) -> mx.array:
        latents = mx.reshape(latents, (1, width // 16, height // 16, 16, 2, 2))
        latents = mx.transpose(latents, (0, 3, 1, 4, 2, 5))
        latents = mx.reshape(latents, (1, 16, width // 16 * 2, height // 16 * 2))
        return latents
    

class Flux1Img2Img(Flux1):

    def __init__(self, repo_id: str, max_sequence_length: int = 512):
        super().__init__(repo_id, max_sequence_length)

    def generate_image(self, seed: int, base_image: PIL.Image.Image, prompt: str, config: Config, strength: float) -> mx.array:
        if config.height != base_image.height or config.width != base_image.width:
            log.warning("Image dimensions do not match config. Overwriting.")
            config.height = base_image.height
            config.width = base_image.width

        if strength < 0.0 or strength > 1.0:
            log.warning("Strength should be between 0.0 and 1.0. Clipping.")
            strength = max(0.0, min(1.0, strength))
        noise = LatentCreator.create(config.height, config.width, seed)
        base_image = ImageUtil.to_array(base_image)
        image_latents = self.vae.encode(base_image)
        image_latents = self._pack_latents(image_latents, config.height, config.width)
        init_timestep = int(config.num_inference_steps * strength)
        config.inference_steps = config.inference_steps[init_timestep:]
        latents = config.sigmas[init_timestep] * noise + (1.0 - config.sigmas[init_timestep]) * image_latents

        return self._generate_from_latents(latents, prompt, config)

