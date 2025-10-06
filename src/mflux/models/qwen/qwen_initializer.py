from mflux.config.model_config import ModelConfig
from mflux.models.common.lora.download.lora_huggingface_downloader import LoRAHuggingFaceDownloader
from mflux.models.common.lora.mapping.lora_loader import LoRALoader
from mflux.models.qwen.model.qwen_text_encoder.qwen_text_encoder import QwenTextEncoder
from mflux.models.qwen.model.qwen_transformer.qwen_transformer import QwenTransformer
from mflux.models.qwen.model.qwen_vae.qwen_vae import QwenVAE
from mflux.models.qwen.tokenizer.qwen_tokenizer_handler import QwenTokenizerHandler
from mflux.models.qwen.weights.qwen_weight_handler import QwenWeightHandler
from mflux.models.qwen.weights.qwen_lora_mapping import QwenLoRAMapping
from mflux.models.qwen.weights.qwen_weight_util import QwenWeightUtil


class QwenImageInitializer:
    @staticmethod
    def init(
        qwen_model,
        model_config: ModelConfig,
        quantize: int | None,
        local_path: str | None,
        lora_paths: list[str] | None = None,
        lora_scales: list[float] | None = None,
        lora_names: list[str] | None = None,
        lora_repo_id: str | None = None,
    ) -> None:
        # 0. Set paths, configs, and prompt_cache for later
        qwen_model.prompt_cache = {}
        qwen_model.model_config = model_config

        # 1. Load the regular weights
        weights = QwenWeightHandler.load_regular_weights(
            repo_id=model_config.model_name,
            local_path=local_path,
        )

        # 2. Initialize tokenizers
        tokenizer_handler = QwenTokenizerHandler(
            repo_id=model_config.model_name,
            local_path=local_path,
        )
        qwen_model.qwen_tokenizer = tokenizer_handler.qwen

        # 3. Initialize all models
        qwen_model.vae = QwenVAE()
        qwen_model.transformer = QwenTransformer()
        qwen_model.text_encoder = QwenTextEncoder()

        # 4. Apply weights and quantize the models
        qwen_model.bits = QwenWeightUtil.set_weights_and_quantize(
            quantize_arg=quantize,
            weights=weights,
            vae=qwen_model.vae,
            transformer=qwen_model.transformer,
            text_encoder=qwen_model.text_encoder,
        )

        # 5. Set LoRA weights
        hf_lora_paths = LoRAHuggingFaceDownloader.download_loras(
            lora_names=lora_names,
            repo_id=lora_repo_id,
            model_name="Qwen",
        )
        qwen_model.lora_paths = (lora_paths or []) + hf_lora_paths
        qwen_model.lora_scales = (lora_scales or []) + [1.0] * len(hf_lora_paths)
        if qwen_model.lora_paths:
            LoRALoader.load_and_apply_lora(
                lora_mapping=QwenLoRAMapping.get_mapping(),
                transformer=qwen_model.transformer,
                lora_files=qwen_model.lora_paths,
                lora_scales=qwen_model.lora_scales,
            )
        else:
            print("⚠️  No LoRA paths provided, skipping LoRA setup")

        print("🔧 === END QWEN LORA WEIGHT SETUP DEBUG ===\n")
