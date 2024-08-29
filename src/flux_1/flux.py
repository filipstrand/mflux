from pathlib import Path

import PIL
import mlx.core as mx
from PIL import Image
from mlx import nn
from mlx.utils import tree_flatten
from tqdm import tqdm

from flux_1.config.config import Config
from flux_1.config.model_config import ModelConfig
from flux_1.config.runtime_config import RuntimeConfig
from flux_1.models.text_encoder.clip_encoder.clip_encoder import CLIPEncoder
from flux_1.models.text_encoder.t5_encoder.t5_encoder import T5Encoder
from flux_1.models.transformer.transformer import Transformer
from flux_1.models.vae.vae import VAE
from flux_1.post_processing.image_util import ImageUtil
from flux_1.tokenizer.clip_tokenizer import TokenizerCLIP
from flux_1.tokenizer.t5_tokenizer import TokenizerT5
from flux_1.tokenizer.tokenizer_handler import TokenizerHandler
from flux_1.weights.weight_handler import WeightHandler


class Flux1:

    def __init__(
            self,
            model_config: ModelConfig,
            bits: int | None = None,
            path: str | None = None,
            is_huggingface: bool = True,
    ):
        self.model_config = model_config

        # Initialize the tokenizers
        tokenizers = TokenizerHandler.load_from_cache_or_huggingface(model_config.model_name, self.model_config.max_sequence_length, path)
        self.t5_tokenizer = TokenizerT5(tokenizers.t5, max_length=self.model_config.max_sequence_length)
        self.clip_tokenizer = TokenizerCLIP(tokenizers.clip)

        # Initialize the models
        weights = Flux1._load_model_weights(model_config.model_name, path, is_huggingface)
        self.vae = VAE(weights.vae)
        self.transformer = Transformer(weights.transformer)
        self.t5_text_encoder = T5Encoder(weights.t5_encoder)
        self.clip_text_encoder = CLIPEncoder(weights.clip_encoder)

        # Optionally quantize the model at initialization
        if bits:
            nn.quantize(self.vae, class_predicate=lambda _, m: isinstance(m, nn.Linear), group_size=64, bits=bits)
            nn.quantize(self.transformer, class_predicate=lambda _, m: isinstance(m, nn.Linear) and len(m.weight[1]) > 64, group_size=64, bits=bits)
            nn.quantize(self.t5_text_encoder, class_predicate=lambda _, m: isinstance(m, nn.Linear), group_size=64, bits=bits)
            nn.quantize(self.clip_text_encoder, class_predicate=lambda _, m: isinstance(m, nn.Linear), group_size=64, bits=bits)

    def generate_image(self, seed: int, prompt: str, config: Config = Config()) -> PIL.Image.Image:
        # Create a new runtime config based on the model type and input parameters
        config = RuntimeConfig(config, self.model_config)

        # 1. Create the initial latents
        latents = mx.random.normal(
            shape=[1, (config.height // 16) * (config.width // 16), 64],
            key=mx.random.key(seed)
        )

        # 2. Embedd the prompt
        t5_tokens = self.t5_tokenizer.tokenize(prompt)
        clip_tokens = self.clip_tokenizer.tokenize(prompt)
        prompt_embeds = self.t5_text_encoder.forward(t5_tokens)
        pooled_prompt_embeds = self.clip_text_encoder.forward(clip_tokens)

        for t in tqdm(range(config.num_inference_steps)):
            # 3.t Predict the noise
            noise = self.transformer.predict(
                t=t,
                prompt_embeds=prompt_embeds,
                pooled_prompt_embeds=pooled_prompt_embeds,
                hidden_states=latents,
                config=config,
            )

            # 4.t Take one denoise step
            dt = config.sigmas[t + 1] - config.sigmas[t]
            latents += noise * dt

            # Evaluate to enable progress tracking
            mx.eval(latents)

        # 5. Decode the latent array
        latents = Flux1._unpack_latents(latents, config.height, config.width)
        decoded = self.vae.decode(latents)
        return ImageUtil.to_image(decoded)

    @staticmethod
    def _unpack_latents(latents: mx.array, height: int, width: int) -> mx.array:
        latents = mx.reshape(latents, (1, width // 16, height // 16, 16, 2, 2))
        latents = mx.transpose(latents, (0, 3, 1, 4, 2, 5))
        latents = mx.reshape(latents, (1, 16, width // 16 * 2, height // 16 * 2))
        return latents

    @staticmethod
    def from_alias(alias: str, bits: int | None = None) -> "Flux1":
        return Flux1(
            model_config=ModelConfig.from_alias(alias),
            bits=bits,
            path=None,
            is_huggingface=True,
        )

    @staticmethod
    def from_disk(model_config: ModelConfig, path: str, bits: int | None = None):
        return Flux1(
            model_config=model_config,
            bits=bits,
            path=path,
            is_huggingface=True,
        )

    @staticmethod
    def from_disk_mlx(model_config: ModelConfig, path: str, bits: int | None = None):
        return Flux1(
            model_config=model_config,
            bits=bits,
            path=path,
            is_huggingface=False,
        )

    @staticmethod
    def _load_model_weights(
            model_name: str,
            path: str | None = None,
            is_huggingface: bool = True,
    ) -> WeightHandler:
        if path is None:
            return WeightHandler.load_from_cache_or_huggingface(model_name)
        else:
            if is_huggingface:
                return WeightHandler.load_huggingface_model_from_disk(path)
            else:
                return WeightHandler.load_quantized_model_from_disk(path)

    def save_model_weights(self, base_path: str):
        def save_weights(model, subdir: str):
            path = Path(base_path) / subdir
            path.mkdir(parents=True, exist_ok=True)
            weights = Flux1._split_weights(dict(tree_flatten(model.parameters())))
            for i, weight in enumerate(weights):
                mx.save_safetensors(str(path / f"{i}.safetensors"), weight)

        # Save each model component
        save_weights(self.vae, "vae")
        save_weights(self.transformer, "transformer")
        save_weights(self.clip_text_encoder, "text_encoder")
        save_weights(self.t5_text_encoder, "text_encoder_2")

    @staticmethod
    def _split_weights(weights: dict, max_file_size_gb: int = 2) -> list:
        # Copied from mlx-examples repo
        max_file_size_bytes = max_file_size_gb << 30
        shards = []
        shard, shard_size = {}, 0
        for k, v in weights.items():
            if shard_size + v.nbytes > max_file_size_bytes:
                shards.append(shard)
                shard, shard_size = {}, 0
            shard[k] = v
            shard_size += v.nbytes
        shards.append(shard)
        return shards
