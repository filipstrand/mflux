from pathlib import Path

from transformers import Qwen2TokenizerFast

from mflux.config.model_config import ModelConfig
from mflux.models.qwen.model.qwen_text_encoder.qwen_vision_language_encoder import QwenVisionLanguageEncoder
from mflux.models.qwen.qwen_initializer import QwenImageInitializer
from mflux.models.qwen.tokenizer.qwen_vision_language_processor import QwenVisionLanguageProcessor
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
            model_config=model_config,
        )

    @staticmethod
    def _init_vision_language_components(
        qwen_model,
        repo_id: str,
        local_path: str | None,
        model_config: ModelConfig | None = None,
    ) -> None:
        # 1. Download or get cached tokenizer
        root_path = Path(local_path) if local_path else QwenImageEditInitializer._download_vl_processor(repo_id)

        # 2. Load only the tokenizer (we implement image processing ourselves)
        tokenizer = QwenImageEditInitializer._load_tokenizer(root_path, repo_id)

        # 3. Create our MLX processor with the tokenizer
        processor = QwenVisionLanguageProcessor(tokenizer=tokenizer)

        # 4. Initialize vision-language tokenizer wrapper
        qwen_model.qwen_vl_tokenizer = QwenVisionLanguageTokenizer(
            processor=processor,
            max_length=1024,
            use_picture_prefix=True,
        )

        # 5. Initialize vision-language encoder (integrated approach like Diffusers)
        qwen_model.qwen_vl_encoder = QwenVisionLanguageEncoder(encoder=qwen_model.text_encoder.encoder)

    @staticmethod
    def _load_tokenizer(root_path: Path, repo_id: str) -> Qwen2TokenizerFast:
        """Load the tokenizer from local path or download from HuggingFace."""
        tokenizer_path = root_path / "tokenizer"
        if not tokenizer_path.exists():
            tokenizer_path = root_path

        try:
            return Qwen2TokenizerFast.from_pretrained(
                pretrained_model_name_or_path=tokenizer_path,
                local_files_only=True,
            )
        except OSError:
            return Qwen2TokenizerFast.from_pretrained(repo_id)

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
