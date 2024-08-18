import PIL
import mlx.core as mx
from PIL import Image
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


class Flux1:

    def __init__(self, repo_id: str):
        self.is_dev = "FLUX.1-dev" in repo_id
        # max_t5_length = 512 if self.is_dev else 256
        max_t5_length = 256
        tokenizers = TokenizerHandler.load_from_disk_or_huggingface(repo_id, max_t5_length)
        self.t5_tokenizer = TokenizerT5(tokenizers.t5, max_length=max_t5_length)
        self.clip_tokenizer = TokenizerCLIP(tokenizers.clip)

        weights = WeightHandler.load_from_disk_or_huggingface(repo_id)
        self.vae = VAE(weights.vae)
        self.transformer = Transformer(weights.transformer)
        self.t5_text_encoder = T5Encoder(weights.t5_encoder)
        self.clip_text_encoder = CLIPEncoder(weights.clip_encoder)

    def generate_image(self, seed: int, prompt: str, config: Config = Config()) -> PIL.Image.Image:
        if self.is_dev:
            config.shift_sigmas()
        latents = LatentCreator.create(config.height, config.width, seed)

        t5_tokens = self.t5_tokenizer.tokenize(prompt)
        clip_tokens = self.clip_tokenizer.tokenize(prompt)
        prompt_embeds = self.t5_text_encoder.forward(t5_tokens)
        pooled_prompt_embeds = self.clip_text_encoder.forward(clip_tokens)

        for t in tqdm(range(config.num_inference_steps)):
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

    @staticmethod
    def _unpack_latents(latents: mx.array, width: int, height: int) -> mx.array:
        latents = mx.reshape(latents, (1, height//16, width//16, 16, 2, 2))
        latents = mx.transpose(latents, (0, 3, 1, 4, 2, 5))
        latents = mx.reshape(latents, (1, 16, height//16 *2, width//16 * 2))
        return latents

    def encode(self, path: str) -> mx.array:
        array = ImageUtil.to_array(Image.open(path))
        return self.vae.encode(array)

    def decode(self, code: mx.array) -> PIL.Image.Image:
        decoded = self.vae.decode(code)
        return ImageUtil.to_image(decoded)
