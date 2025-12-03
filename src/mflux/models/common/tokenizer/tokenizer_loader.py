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
            chat_template=definition.chat_template,
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
        chat_template: str | None = None,
    ):
        if hasattr(transformers, tokenizer_class):
            cls = getattr(transformers, tokenizer_class)
        else:
            raise ValueError(f"Unknown tokenizer class: {tokenizer_class}")

        # Workaround for Qwen2Tokenizer bug in transformers 5.0.0rc0
        # The tokenizer doesn't properly load vocab/merges from files, need to pass them directly
        if tokenizer_class == "Qwen2Tokenizer":
            tokenizer = TokenizerLoader._load_qwen2_tokenizer_workaround(tokenizer_path, cls)
        else:
            tokenizer = cls.from_pretrained(
                pretrained_model_name_or_path=str(tokenizer_path),
                local_files_only=True,
            )

        # Apply chat_template from definition if provided and not already set
        if chat_template and not getattr(tokenizer, "chat_template", None):
            tokenizer.chat_template = chat_template

        return tokenizer

    @staticmethod
    def _load_qwen2_tokenizer_workaround(tokenizer_path: Path, cls):
        """Load Qwen2Tokenizer with explicit vocab and merges data.

        In transformers 5.0, Qwen2Tokenizer builds its internal BPE tokenizer from the
        `vocab` and `merges` parameters BEFORE calling super().__init__(). The `vocab_file`
        and `merges_file` parameters are only passed to the parent class for saving purposes
        and are NOT automatically loaded. We must load and parse these files ourselves.

        We also need to load special tokens from tokenizer_config.json to ensure tokens like
        <|im_start|>, <|im_end|>, <|vision_start|>, etc. are properly registered.
        """
        import json

        from tokenizers import AddedToken

        vocab_file = tokenizer_path / "vocab.json"
        merges_file = tokenizer_path / "merges.txt"
        config_file = tokenizer_path / "tokenizer_config.json"

        if not vocab_file.exists() or not merges_file.exists():
            # Fall back to standard loading if files don't exist
            return cls.from_pretrained(
                pretrained_model_name_or_path=str(tokenizer_path),
                local_files_only=True,
            )

        # Load vocab as dict[str, int]
        with open(vocab_file, encoding="utf-8") as f:
            vocab = json.load(f)

        # Load merges as list of tuples (pair of strings)
        # The merges.txt file has format: "token1 token2" per line (skip header if present)
        merges = []
        with open(merges_file, encoding="utf-8") as f:
            for line in f:
                line = line.rstrip()
                # Skip empty lines and the optional version header (e.g., "#version: 0.2")
                if not line or line.startswith("#"):
                    continue
                parts = line.split()
                if len(parts) == 2:
                    merges.append((parts[0], parts[1]))

        # Load tokenizer config for special tokens and chat template
        config_kwargs = {}
        chat_template = None
        if config_file.exists():
            with open(config_file, encoding="utf-8") as f:
                config = json.load(f)

            # Extract added_tokens_decoder to add special tokens
            added_tokens_decoder = config.get("added_tokens_decoder", {})
            if added_tokens_decoder:
                # Convert to the format expected by the tokenizer
                config_kwargs["added_tokens_decoder"] = {
                    int(k): AddedToken(
                        content=v["content"],
                        lstrip=v.get("lstrip", False),
                        rstrip=v.get("rstrip", False),
                        single_word=v.get("single_word", False),
                        normalized=v.get("normalized", False),
                        special=v.get("special", True),
                    )
                    for k, v in added_tokens_decoder.items()
                }

            # Extract chat_template if present
            chat_template = config.get("chat_template")

        tokenizer = cls(vocab=vocab, merges=merges, **config_kwargs)

        # Set chat_template after initialization (not a constructor param)
        if chat_template:
            tokenizer.chat_template = chat_template

        return tokenizer

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
