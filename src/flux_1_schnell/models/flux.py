import PIL
import mlx.core as mx
from PIL import Image

from flux_1_schnell.models.text_encoder.clip_encoder.clip_encoder import CLIPEncoder
from flux_1_schnell.tokenizer.clip_tokenizer import TokenizerCLIP
from flux_1_schnell.models.text_encoder.t5_encoder.t5_encoder import T5Encoder
from flux_1_schnell.tokenizer.t5_tokenizer import TokenizerT5

from flux_1_schnell.config.config import Config
from flux_1_schnell.latent_creator.latent_creator import LatentCreator
from flux_1_schnell.models.text_encoder.text_encoder import TextEncoder
from flux_1_schnell.models.transformer.transformer import Transformer
from flux_1_schnell.models.vae.vae import VAE
from flux_1_schnell.post_processing.image_util import ImageUtil
from flux_1_schnell.scheduler.scheduler import FlowMatchEulerDiscreteNoiseScheduler
from flux_1_schnell.tokenizer.tokenizer_handler import TokenizerHandler
from flux_1_schnell.weights.weight_handler import WeightHandler


class Flux1Schnell:

    def __init__(self, root_path: str):
        tokenizers = TokenizerHandler.load_from_disk_via_huggingface_transformers(root_path)
        self.clip_tokenizer = TokenizerCLIP(tokenizers.clip)
        self.t5_tokenizer = TokenizerT5(tokenizers.t5)

        weights = WeightHandler.load_from_disk(root_path)
        self.vae = VAE(weights.vae)
        self.transformer = Transformer(weights.transformer)
        self.clip_text_encoder = CLIPEncoder(weights.clip_encoder)
        self.t5_text_encoder = T5Encoder(weights.t5_encoder)

    def generate_image(self, seed: int, prompt: str, config: Config = Config()) -> PIL.Image.Image:
        latents = LatentCreator.create(seed)
        prompt_embeds, pooled_prompt_embeds = TextEncoder.encode(
            prompt=prompt,
            clip_tokenizer=self.clip_tokenizer,
            t5_tokenizer=self.t5_tokenizer,
            clip_text_encoder=self.clip_text_encoder,
            t5_text_encoder=self.t5_text_encoder
        )

        for t in range(config.num_inference_steps):
            noise = self.transformer.predict(
                t=t,
                prompt_embeds=prompt_embeds,
                pooled_prompt_embeds=pooled_prompt_embeds,
                hidden_states=latents,
                config=config
            )

            latents = FlowMatchEulerDiscreteNoiseScheduler.denoise(
                t=t,
                noise=noise,
                latent=latents,
                config=config
            )

        latents = Flux1Schnell._unpack_latents(latents)
        decoded = self.vae.decode(latents)
        return ImageUtil.to_image(decoded)

    @staticmethod
    def _unpack_latents(latents):
        latents = mx.reshape(latents, (1, 64, 64, 16, 2, 2))
        latents = mx.transpose(latents, (0, 3, 1, 4, 2, 5))
        latents = mx.reshape(latents, (1, 16, 128, 128))
        return latents

    def encode(self, path: str) -> mx.array:
        array = ImageUtil.to_array(Image.open(path))
        return self.vae.encode(array)

    def decode(self, code: mx.array) -> PIL.Image.Image:
        decoded = self.vae.decode(code)
        return ImageUtil.to_image(decoded)
