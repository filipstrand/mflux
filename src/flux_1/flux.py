from pathlib import Path

import PIL
import mlx.core as mx
from PIL import Image
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
from mlx import nn
from mlx.utils import tree_flatten
from tqdm import tqdm


class Flux1:

    def __init__(
            self,
            model_config: ModelConfig,
            quantize_full_weights: int | None = None,
            local_path: str | None = None,
    ):
        self.model_config = model_config
        self.quantize_full_weights = quantize_full_weights

        # Load and initialize the tokenizers from disk, huggingface cache, or download from huggingface
        tokenizers = TokenizerHandler(model_config.model_name, self.model_config.max_sequence_length, local_path)
        self.t5_tokenizer = TokenizerT5(tokenizers.t5, max_length=self.model_config.max_sequence_length)
        self.clip_tokenizer = TokenizerCLIP(tokenizers.clip)

        # Initialize the models
        self.vae = VAE()
        self.transformer = Transformer(model_config)
        self.t5_text_encoder = T5Encoder()
        self.clip_text_encoder = CLIPEncoder()

        # Load the weights from disk, huggingface cache, or download from huggingface
        weights = WeightHandler(repo_id=model_config.model_name, local_path=local_path)

        # Set the loaded weights if they are not quantized
        if weights.quantization_level is None:
            self._set_model_weights(weights)

        # Optionally quantize the model here at initialization (also required if about to load quantized weights)
        if quantize_full_weights is not None or weights.quantization_level is not None:
            bits = weights.quantization_level if weights.quantization_level is not None else quantize_full_weights
            nn.quantize(self.vae, class_predicate=lambda _, m: isinstance(m, nn.Linear), group_size=64, bits=bits)
            nn.quantize(self.transformer, class_predicate=lambda _, m: isinstance(m, nn.Linear) and len(m.weight[1]) > 64, group_size=64, bits=bits)
            nn.quantize(self.t5_text_encoder, class_predicate=lambda _, m: isinstance(m, nn.Linear), group_size=64, bits=bits)
            nn.quantize(self.clip_text_encoder, class_predicate=lambda _, m: isinstance(m, nn.Linear), group_size=64, bits=bits)

        # If loading previously saved quantized weights, the weights must be set after modules have been quantized
        if weights.quantization_level is not None:
            self._set_model_weights(weights)

    def _generate_latents(self, seed: int, prompt: str, config: Config):
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

            yield t, latents

    def generate_image(self, seed: int, prompt: str, config: Config = Config()) -> PIL.Image.Image:
        for t, latents in self._generate_latents(seed, prompt, config):
            pass  # All processing happens in the loop inside _generate_latents

        # Decode the latent array after all steps are completed

        latents = Flux1._unpack_latents(latents, config.height, config.width)
        decoded = self.vae.decode(latents)
        return ImageUtil.to_image(decoded)

    def stream_generate_image(self, seed: int, prompt: str, report_step: int = 1, config: Config = Config()):

        for t, latents in self._generate_latents(seed, prompt, config):
            if (t + 1) % report_step == 0:
                current_latents = Flux1._unpack_latents(latents, config.height, config.width)
                decoded_image = self.vae.decode(current_latents)
                current_image = ImageUtil.to_image(decoded_image)
                yield current_image

        final_latents = Flux1._unpack_latents(latents, config.height, config.width)
        final_decoded_image = self.vae.decode(final_latents)
        final_image = ImageUtil.to_image(final_decoded_image)
        yield final_image

    @staticmethod
    def _unpack_latents(latents: mx.array, height: int, width: int) -> mx.array:
        latents = mx.reshape(latents, (1, width // 16, height // 16, 16, 2, 2))
        latents = mx.transpose(latents, (0, 3, 1, 4, 2, 5))
        latents = mx.reshape(latents, (1, 16, width // 16 * 2, height // 16 * 2))
        return latents

    @staticmethod
    def from_alias(alias: str) -> "Flux1":
        return Flux1(ModelConfig.from_alias(alias))

    def _set_model_weights(self, weights):
        self.vae.update(weights.vae)
        self.transformer.update(weights.transformer)
        self.t5_text_encoder.update(weights.t5_encoder)
        self.clip_text_encoder.update(weights.clip_encoder)

    def save_model(self, base_path: str):
        def _save_tokenizer(tokenizer, subdir: str):
            path = Path(base_path) / subdir
            path.mkdir(parents=True, exist_ok=True)
            tokenizer.save_pretrained(path)

        def _save_weights(model, subdir: str):
            path = Path(base_path) / subdir
            path.mkdir(parents=True, exist_ok=True)
            weights = _split_weights(dict(tree_flatten(model.parameters())))
            for i, weight in enumerate(weights):
                mx.save_safetensors(str(path / f"{i}.safetensors"), weight, {"quantization_level": str(self.quantize_full_weights)})

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

        # Save the tokenizers
        _save_tokenizer(self.clip_tokenizer.tokenizer, "tokenizer")
        _save_tokenizer(self.t5_tokenizer.tokenizer, "tokenizer_2")

        # Save the models
        _save_weights(self.vae, "vae")
        _save_weights(self.transformer, "transformer")
        _save_weights(self.clip_text_encoder, "text_encoder")
        _save_weights(self.t5_text_encoder, "text_encoder_2")
