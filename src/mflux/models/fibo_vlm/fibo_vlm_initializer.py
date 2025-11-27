import os
from pathlib import Path

from transformers import Qwen2Tokenizer

from mflux.models.fibo.tokenizer.qwen2vl_processor import Qwen2VLProcessor
from mflux.models.fibo_vlm.model.qwen3_vl_decoder import Qwen3VLDecoder
from mflux.models.fibo_vlm.model.qwen3_vl_vision_model import Qwen3VLVisionModel
from mflux.models.fibo_vlm.weights.fibo_vlm_weight_handler import FIBOVLMWeightHandler
from mflux.utils.download import snapshot_download


class FIBOVLMInitializer:
    @staticmethod
    def init(
        vlm_model,
        model_id: str = "briaai/FIBO-vlm",
        local_path: str | None = None,
    ) -> None:
        # 1. Load VLM weights
        weights = FIBOVLMWeightHandler.load_vlm_regular_weights(
            repo_id=model_id,
            local_path=local_path,
        )

        # 2. Initialize processor for tokenization
        tokenizer = FIBOVLMInitializer._get_tokenizer(local_path, model_id)
        vlm_model.processor = Qwen2VLProcessor(tokenizer=tokenizer)

        # 3. Initialize all models
        vlm_model.decoder = Qwen3VLDecoder(visual=Qwen3VLVisionModel())

        # 4. Apply weights to decoder and visual encoder
        vlm_model.decoder.update(weights.decoder, strict=False)
        vlm_model.decoder.visual.update(weights.visual, strict=False)

        # Store model ID and local path
        vlm_model.model_id = model_id
        vlm_model.local_path = local_path

    @staticmethod
    def _get_tokenizer(local_path, model_id):
        # Get the root path
        if local_path:
            root_path = Path(local_path)
        else:
            root_path = FIBOVLMInitializer._get_model_path(model_id)

        # Try different possible tokenizer paths (like FiboTokenizerHandler does)
        tokenizer_path = root_path / "tokenizer"
        if not tokenizer_path.exists():
            tokenizer_path = root_path / "text_encoder"
        if not tokenizer_path.exists():
            # Tokenizer files are in the root - this is the case for FIBO-vlm
            tokenizer_path = root_path

        # Set HF_HUB_OFFLINE to force offline mode
        old_offline = os.environ.get("HF_HUB_OFFLINE")
        try:
            os.environ["HF_HUB_OFFLINE"] = "1"
            # Use Qwen2Tokenizer directly instead of AutoTokenizer to avoid config confusion
            tokenizer = Qwen2Tokenizer.from_pretrained(
                pretrained_model_name_or_path=str(tokenizer_path),
                local_files_only=True,
            )
        finally:
            # Restore original value
            if old_offline is None:
                os.environ.pop("HF_HUB_OFFLINE", None)
            else:
                os.environ["HF_HUB_OFFLINE"] = old_offline

        return tokenizer

    @staticmethod
    def _get_model_path(model_id: str) -> Path:
        # Try to use cached path first
        try:
            root_path = Path(
                snapshot_download(
                    repo_id=model_id,
                    local_files_only=True,
                )
            )
            # Check if tokenizer files actually exist - the model weights might be cached
            # but tokenizer files may not have been downloaded yet
            if not FIBOVLMInitializer._tokenizer_files_exist(root_path):
                # Tokenizer files missing, need to download them
                root_path = Path(
                    snapshot_download(
                        repo_id=model_id,
                        local_files_only=False,
                    )
                )
        except (FileNotFoundError, OSError):
            # Model not in cache, allow download if online
            root_path = Path(
                snapshot_download(
                    repo_id=model_id,
                )
            )
        return root_path

    @staticmethod
    def _tokenizer_files_exist(root_path: Path) -> bool:
        # Check for vocab.json in possible tokenizer locations
        possible_paths = [
            root_path / "tokenizer" / "vocab.json",
            root_path / "text_encoder" / "vocab.json",
            root_path / "vocab.json",
        ]
        return any(p.exists() for p in possible_paths)
