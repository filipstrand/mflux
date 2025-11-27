from pathlib import Path
from typing import TYPE_CHECKING

import transformers
from huggingface_hub import snapshot_download

from mflux.models.common.tokenizer.tokenizer import (
    BaseTokenizer,
    LanguageTokenizer,
    VisionLanguageTokenizer,
)

if TYPE_CHECKING:
    from mflux.models.common.weights.loading.weight_definition import TokenizerDefinition


class TokenizerLoader:
    @staticmethod
    def load(
        definition: "TokenizerDefinition",
        model_path: str,
    ) -> BaseTokenizer:
        tokenizer_path = TokenizerLoader._resolve_path(
            model_path=model_path,
            hf_subdir=definition.hf_subdir,
            fallback_subdirs=definition.fallback_subdirs,
            download_patterns=definition.download_patterns,
        )

        raw_tokenizer = TokenizerLoader._load_raw_tokenizer(
            tokenizer_path=tokenizer_path,
            tokenizer_class=definition.tokenizer_class,
        )

        return TokenizerLoader._create_tokenizer(
            raw_tokenizer=raw_tokenizer,
            definition=definition,
        )

    @staticmethod
    def load_all(
        definitions: list["TokenizerDefinition"],
        model_path: str,
        max_length_overrides: dict[str, int] | None = None,
    ) -> dict[str, BaseTokenizer]:
        max_length_overrides = max_length_overrides or {}
        result = {}
        for d in definitions:
            tokenizer = TokenizerLoader.load(d, model_path)
            if d.name in max_length_overrides:
                tokenizer.max_length = max_length_overrides[d.name]
            result[d.name] = tokenizer
        return result

    @staticmethod
    def _resolve_path(
        model_path: str,
        hf_subdir: str,
        fallback_subdirs: list[str] | None,
        download_patterns: list[str] | None,
    ) -> Path:
        expanded = Path(model_path).expanduser()
        if expanded.exists():
            root_path = expanded
        elif "/" in model_path and model_path.count("/") == 1 and not model_path.startswith(("./", "../")):
            patterns = download_patterns or [f"{hf_subdir}/**"]
            root_path = Path(
                snapshot_download(
                    repo_id=model_path,
                    allow_patterns=patterns,
                )
            )
        else:
            raise FileNotFoundError(
                f"Model not found: '{model_path}'. "
                f"If local path, make sure it exists. "
                f"If HuggingFace repo, use 'org/model' format."
            )

        tokenizer_path = root_path / hf_subdir
        if tokenizer_path.exists():
            return tokenizer_path

        if fallback_subdirs:
            for subdir in fallback_subdirs:
                if subdir == ".":
                    if TokenizerLoader._has_tokenizer_files(root_path):
                        return root_path
                else:
                    fallback_path = root_path / subdir
                    if fallback_path.exists():
                        return fallback_path

        return tokenizer_path

    @staticmethod
    def _has_tokenizer_files(path: Path) -> bool:
        tokenizer_indicators = ["vocab.json", "tokenizer.json", "tokenizer_config.json"]
        return any((path / f).exists() for f in tokenizer_indicators)

    @staticmethod
    def _load_raw_tokenizer(
        tokenizer_path: Path,
        tokenizer_class: str,
    ):
        if hasattr(transformers, tokenizer_class):
            cls = getattr(transformers, tokenizer_class)
        else:
            raise ValueError(f"Unknown tokenizer class: {tokenizer_class}")

        return cls.from_pretrained(
            pretrained_model_name_or_path=str(tokenizer_path),
            local_files_only=True,
        )

    @staticmethod
    def _create_tokenizer(
        raw_tokenizer,
        definition: "TokenizerDefinition",
    ) -> BaseTokenizer:
        encoder_class = definition.encoder_class

        if encoder_class is VisionLanguageTokenizer:
            if definition.processor_class is None:
                raise ValueError("VisionLanguageTokenizer requires processor_class in definition")
            processor = definition.processor_class(tokenizer=raw_tokenizer)
            return VisionLanguageTokenizer(
                tokenizer=raw_tokenizer,
                processor=processor,
                max_length=definition.max_length,
                template=definition.template,
                image_token=definition.image_token,
            )
        else:
            # Default to LanguageTokenizer for all text-only cases
            return LanguageTokenizer(
                tokenizer=raw_tokenizer,
                max_length=definition.max_length,
                padding=definition.padding,
                template=definition.template,
                use_chat_template=definition.use_chat_template,
                chat_template_kwargs=definition.chat_template_kwargs or {},
                add_special_tokens=definition.add_special_tokens,
            )
