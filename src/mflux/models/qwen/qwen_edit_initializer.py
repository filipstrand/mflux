from pathlib import Path

from transformers import Qwen2VLProcessor

from mflux.config.model_config import ModelConfig
# Note: No longer need custom vision components - using integrated text encoder approach
from mflux.models.qwen.qwen_initializer import QwenImageInitializer
from mflux.models.qwen.tokenizer.qwen_vision_language_tokenizer import QwenVisionLanguageTokenizer
from mflux.utils.download import snapshot_download


class QwenImageEditInitializer:
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
        # 1. Initialize the base Qwen Image model (VAE, transformer, text encoder, etc.)
        QwenImageInitializer.init(
            qwen_model=qwen_model,
            model_config=model_config,
            quantize=quantize,
            local_path=local_path,
            lora_paths=lora_paths,
            lora_scales=lora_scales,
            lora_names=lora_names,
            lora_repo_id=lora_repo_id,
        )

        # 2. Add vision-language components for edit functionality
        QwenImageEditInitializer._init_vision_language_components(
            qwen_model=qwen_model,
            repo_id=model_config.model_name,
            local_path=local_path,
        )

        # 3. Apply quantization to vision-language components if requested
        # Skip VL encoder quantization as it contains visual components with incompatible dimensions
        if quantize is not None and hasattr(qwen_model, 'qwen_vl_encoder'):
            print(f"🔧 Skipping VL encoder quantization (contains visual components with incompatible dimensions)")

    @staticmethod
    def _init_vision_language_components(
        qwen_model,
        repo_id: str,
        local_path: str | None,
    ) -> None:
        # 1. Download or get cached vision-language processor
        root_path = Path(local_path) if local_path else QwenImageEditInitializer._download_vl_processor(repo_id)

        # 2. Load Qwen2VLProcessor from the SAME model as our weights (Qwen-Image-Edit)
        processor_path = root_path / "processor"
        if not processor_path.exists():
            processor_path = root_path

        try:
            print(f"🔧 Loading processor from {processor_path} (matching our weights)")
            processor = Qwen2VLProcessor.from_pretrained(
                pretrained_model_name_or_path=processor_path,
                local_files_only=True,
            )
        except OSError as e:
            print(f"🔧 Failed to load processor from local path: {e}")
            print(f"🔧 Falling back to loading processor from {repo_id}")
            processor = Qwen2VLProcessor.from_pretrained(repo_id)

        # 3. Initialize vision-language tokenizer (HuggingFace is fine for tokenization)
        qwen_model.qwen_vl_tokenizer = QwenVisionLanguageTokenizer(
            processor=processor,
            max_length=1024,
        )

        # 4. Initialize vision-language encoder (integrated approach like Diffusers)
        # Create VL encoder that wraps the text encoder
        from mflux.models.qwen.model.qwen_text_encoder.qwen_vision_language_encoder import QwenVisionLanguageEncoder
        qwen_model.qwen_vl_encoder = QwenVisionLanguageEncoder(encoder=qwen_model.text_encoder.encoder)

        # Note: Visual processing should be handled by the integrated text encoder model
        # The key missing piece is that our QwenEncoder needs to support pixel_values and image_grid_thw parameters
        # like Qwen2_5_VLForConditionalGeneration does in Diffusers

    @staticmethod
    def _download_vl_processor(repo_id: str) -> Path:
        return Path(
            snapshot_download(
                repo_id=repo_id,
                allow_patterns=[
                    "processor/**",
                    "preprocessor_config.json",
                    "tokenizer/**",
                    "added_tokens.json",
                    "chat_template.jinja",
                    "tokenizer.json",
                    "tokenizer_config.json",
                    "vocab.json",
                    "merges.txt",
                ],
            )
        )
