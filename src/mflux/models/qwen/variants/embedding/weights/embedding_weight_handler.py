"""Weight loading for Qwen3-VL embedding and reranker models.

Handles loading weights from HuggingFace format and converting to MLX.
"""

import logging
from pathlib import Path
from typing import Any

import mlx.core as mx

logger = logging.getLogger(__name__)


class EmbeddingWeightHandler:
    """Handles weight loading for embedding and reranker models.

    Loads weights from HuggingFace safetensors format and maps them
    to the MLX model structure.
    """

    # Token IDs for reranker score weight
    YES_TOKEN_ID = None  # Will be determined from tokenizer
    NO_TOKEN_ID = None

    def __init__(self, model_name_or_path: str):
        """Initialize the weight handler.

        Args:
            model_name_or_path: HuggingFace model ID or local path
        """
        self.model_name_or_path = model_name_or_path
        self._weight_files = None
        self._weights = None

    def load_weights(
        self,
        model: Any,
        quantize_vision: bool = False,
        is_reranker: bool = False,
    ) -> None:
        """Load weights into the model.

        Args:
            model: The MLX model to load weights into
            quantize_vision: Whether to quantize vision encoder
            is_reranker: Whether this is a reranker model
        """
        logger.info(f"Loading weights from {self.model_name_or_path}")

        # Download/locate weight files
        weight_files = self._get_weight_files()

        # Load all weights
        weights = self._load_safetensors(weight_files)

        # Map weights to model
        self._map_weights_to_model(model, weights, quantize_vision)

        # Handle reranker-specific weights
        if is_reranker:
            self._load_score_weight(model, weights)

        logger.info("Weights loaded successfully")

    def _get_weight_files(self) -> list[Path]:
        """Get the weight file paths."""
        if self._weight_files is not None:
            return self._weight_files

        path = Path(self.model_name_or_path)

        if path.exists():
            # Local path
            self._weight_files = list(path.glob("*.safetensors"))
        else:
            # HuggingFace model ID - download
            try:
                from huggingface_hub import snapshot_download

                local_dir = snapshot_download(
                    self.model_name_or_path,
                    allow_patterns=["*.safetensors", "*.json"],
                )
                self._weight_files = list(Path(local_dir).glob("*.safetensors"))
            except ImportError:
                raise ImportError(
                    "huggingface_hub required for downloading models. Install with: pip install huggingface_hub"
                )

        if not self._weight_files:
            raise ValueError(f"No safetensors files found at {self.model_name_or_path}")

        return self._weight_files

    def _load_safetensors(self, files: list[Path]) -> dict[str, mx.array]:
        """Load weights from safetensors files."""
        if self._weights is not None:
            return self._weights

        weights = {}

        for file in files:
            logger.debug(f"Loading {file}")
            file_weights = mx.load(str(file))
            weights.update(file_weights)

        self._weights = weights
        logger.info(f"Loaded {len(weights)} weight tensors")
        return weights

    def _map_weights_to_model(
        self,
        model: Any,
        weights: dict[str, mx.array],
        quantize_vision: bool,
    ) -> None:
        """Map HuggingFace weights to MLX model structure.

        Qwen3-VL uses prefixes:
        - model.language_model.embed_tokens.weight
        - model.language_model.layers.{n}.*
        - model.language_model.norm.weight
        - model.visual.*
        """
        # Get the encoder from the model
        encoder = model.encoder

        # Detect weight prefix (Qwen3-VL uses model.language_model.*)
        if "model.language_model.embed_tokens.weight" in weights:
            lang_prefix = "model.language_model"
            vision_prefix = "model.visual"
        elif "model.embed_tokens.weight" in weights:
            lang_prefix = "model"
            vision_prefix = "visual"
        else:
            raise ValueError("Unknown weight format - cannot find embed_tokens.weight")

        loaded_count = 0
        skipped_count = 0

        # Embed tokens
        embed_key = f"{lang_prefix}.embed_tokens.weight"
        if embed_key in weights:
            encoder.embed_tokens.weight = weights[embed_key]
            loaded_count += 1

        # Final norm
        norm_key = f"{lang_prefix}.norm.weight"
        if norm_key in weights:
            encoder.norm.weight = weights[norm_key]
            loaded_count += 1

        # Transformer layers
        for layer_idx, layer in enumerate(encoder.layers):
            prefix = f"{lang_prefix}.layers.{layer_idx}"

            # Layer norms
            if f"{prefix}.input_layernorm.weight" in weights:
                layer.input_layernorm.weight = weights[f"{prefix}.input_layernorm.weight"]
                loaded_count += 1
            if f"{prefix}.post_attention_layernorm.weight" in weights:
                layer.post_attention_layernorm.weight = weights[f"{prefix}.post_attention_layernorm.weight"]
                loaded_count += 1

            # Attention (Qwen3-VL has no bias, has QK norms)
            attn = layer.self_attn
            attn_mappings = [
                ("q_proj.weight", "q_proj", "weight"),
                ("k_proj.weight", "k_proj", "weight"),
                ("v_proj.weight", "v_proj", "weight"),
                ("o_proj.weight", "o_proj", "weight"),
            ]

            for hf_name, attr_name, weight_name in attn_mappings:
                hf_key = f"{prefix}.self_attn.{hf_name}"
                if hf_key in weights:
                    setattr(getattr(attn, attr_name), weight_name, weights[hf_key])
                    loaded_count += 1

            # QK normalization (Qwen3-VL specific)
            if hasattr(attn, "q_norm") and f"{prefix}.self_attn.q_norm.weight" in weights:
                attn.q_norm.weight = weights[f"{prefix}.self_attn.q_norm.weight"]
                loaded_count += 1
            if hasattr(attn, "k_norm") and f"{prefix}.self_attn.k_norm.weight" in weights:
                attn.k_norm.weight = weights[f"{prefix}.self_attn.k_norm.weight"]
                loaded_count += 1

            # MLP
            mlp = layer.mlp
            mlp_mappings = [
                ("gate_proj.weight", "gate_proj", "weight"),
                ("up_proj.weight", "up_proj", "weight"),
                ("down_proj.weight", "down_proj", "weight"),
            ]

            for hf_name, attr_name, weight_name in mlp_mappings:
                hf_key = f"{prefix}.mlp.{hf_name}"
                if hf_key in weights:
                    setattr(getattr(mlp, attr_name), weight_name, weights[hf_key])
                    loaded_count += 1

        # Vision encoder (load later when init_vision is called)
        # Store vision weights for later loading
        vision_weights = {}
        for key, value in weights.items():
            if key.startswith(f"{vision_prefix}."):
                # Normalize to "visual." prefix for load_vision_weights_to_encoder
                normalized_key = "visual." + key[len(vision_prefix) + 1 :]
                vision_weights[normalized_key] = value
                skipped_count += 1

        # Store vision weights on encoder for later
        encoder._vision_weights = vision_weights
        encoder._quantize_vision = quantize_vision
        # Store embed_tokens for weight-tied lm_head
        encoder._embed_tokens_weight = weights.get(embed_key)

        logger.info(f"Loaded {loaded_count} weights, {skipped_count} vision weights pending")

    def _load_score_weight(
        self,
        model: Any,
        weights: dict[str, mx.array],
    ) -> None:
        """Load the binary classification score weight for reranker.

        The score weight is computed as: lm_head[yes_token] - lm_head[no_token]

        Qwen3-VL uses weight tying (tie_word_embeddings=True), so lm_head
        shares weights with embed_tokens.
        """
        # Try different possible locations for lm_head weights
        lm_head = None
        for key in [
            "lm_head.weight",
            "model.language_model.lm_head.weight",
            # Weight tying: lm_head = embed_tokens
            "model.language_model.embed_tokens.weight",
            "model.embed_tokens.weight",
        ]:
            if key in weights:
                lm_head = weights[key]
                logger.debug(f"Using {key} as lm_head for reranker scoring")
                break

        if lm_head is None:
            raise ValueError(
                "Could not find lm_head or embed_tokens weight for reranker. "
                "Available keys: " + ", ".join(list(weights.keys())[:5]) + "..."
            )

        # Get yes/no token IDs from tokenizer
        try:
            from transformers import AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(self.model_name_or_path)
            yes_id = tokenizer.encode("yes", add_special_tokens=False)[0]
            no_id = tokenizer.encode("no", add_special_tokens=False)[0]
        except (ImportError, OSError, KeyError, IndexError) as e:
            logger.warning(f"Could not get token IDs from tokenizer: {e}")
            # Fallback to common token IDs (Qwen3 vocabulary)
            # These IDs are specific to Qwen3 tokenizers - validate model compatibility
            if not self.model_name_or_path.startswith("Qwen"):
                logger.warning(
                    f"Using hardcoded Qwen token IDs for non-Qwen model '{self.model_name_or_path}'. "
                    "This may produce incorrect reranker scores."
                )
            yes_id = 9454  # "yes" in Qwen3 tokenizer
            no_id = 2152  # "no" in Qwen3 tokenizer

        # Compute score weight: yes - no
        score_weight = lm_head[yes_id] - lm_head[no_id]

        # Set on model
        model.set_score_weight(score_weight)
        logger.info(f"Loaded score weight from lm_head (yes={yes_id}, no={no_id})")


def load_vision_weights_to_encoder(encoder: Any) -> None:
    """Load stored vision weights into the encoder's vision transformer.

    This should be called after encoder.init_vision().

    Qwen3-VL-2B vision architecture:
    - 24 blocks with LayerNorm (not RMSNorm), standard MLP (not SwiGLU)
    - Patch embed: conv3d with bias
    - Block MLP: linear_fc1 -> GELU -> linear_fc2
    - Merger: norm -> linear_fc1 -> GELU -> linear_fc2
    """
    if not hasattr(encoder, "_vision_weights") or encoder._vision_weights is None:
        logger.warning("No vision weights stored - skipping vision weight loading")
        return

    if encoder.visual is None:
        raise RuntimeError("Vision transformer not initialized. Call init_vision() first.")

    weights = encoder._vision_weights
    loaded_count = 0

    visual = encoder.visual

    # Patch embedding (with bias)
    if "visual.patch_embed.proj.weight" in weights:
        # Transpose for MLX conv3d: HF [out, in, D, H, W] -> MLX [out, D, H, W, in]
        w = weights["visual.patch_embed.proj.weight"]
        if len(w.shape) == 5:
            w = mx.transpose(w, (0, 2, 3, 4, 1))
        visual.patch_embed.proj.weight = w
        loaded_count += 1
    if "visual.patch_embed.proj.bias" in weights:
        visual.patch_embed.proj.bias = weights["visual.patch_embed.proj.bias"]
        loaded_count += 1

    # Vision blocks
    for block_idx, block in enumerate(visual.blocks):
        prefix = f"visual.blocks.{block_idx}"

        # Attention QKV
        if f"{prefix}.attn.qkv.weight" in weights:
            block.attn.qkv.weight = weights[f"{prefix}.attn.qkv.weight"]
            loaded_count += 1
        if f"{prefix}.attn.qkv.bias" in weights:
            block.attn.qkv.bias = weights[f"{prefix}.attn.qkv.bias"]
            loaded_count += 1

        # Attention projection
        if f"{prefix}.attn.proj.weight" in weights:
            block.attn.proj.weight = weights[f"{prefix}.attn.proj.weight"]
            loaded_count += 1
        if f"{prefix}.attn.proj.bias" in weights:
            block.attn.proj.bias = weights[f"{prefix}.attn.proj.bias"]
            loaded_count += 1

        # MLP (standard 2-layer: linear_fc1 -> linear_fc2)
        if f"{prefix}.mlp.linear_fc1.weight" in weights:
            block.mlp.linear_fc1.weight = weights[f"{prefix}.mlp.linear_fc1.weight"]
            loaded_count += 1
        if f"{prefix}.mlp.linear_fc1.bias" in weights:
            block.mlp.linear_fc1.bias = weights[f"{prefix}.mlp.linear_fc1.bias"]
            loaded_count += 1
        if f"{prefix}.mlp.linear_fc2.weight" in weights:
            block.mlp.linear_fc2.weight = weights[f"{prefix}.mlp.linear_fc2.weight"]
            loaded_count += 1
        if f"{prefix}.mlp.linear_fc2.bias" in weights:
            block.mlp.linear_fc2.bias = weights[f"{prefix}.mlp.linear_fc2.bias"]
            loaded_count += 1

        # Layer norms (with bias - LayerNorm not RMSNorm)
        if f"{prefix}.norm1.weight" in weights:
            block.norm1.weight = weights[f"{prefix}.norm1.weight"]
            loaded_count += 1
        if f"{prefix}.norm1.bias" in weights:
            block.norm1.bias = weights[f"{prefix}.norm1.bias"]
            loaded_count += 1
        if f"{prefix}.norm2.weight" in weights:
            block.norm2.weight = weights[f"{prefix}.norm2.weight"]
            loaded_count += 1
        if f"{prefix}.norm2.bias" in weights:
            block.norm2.bias = weights[f"{prefix}.norm2.bias"]
            loaded_count += 1

    # Merger (Qwen3-VL-2B uses norm + linear_fc1 + linear_fc2)
    merger = visual.merger
    if "visual.merger.norm.weight" in weights:
        merger.norm.weight = weights["visual.merger.norm.weight"]
        loaded_count += 1
    if "visual.merger.norm.bias" in weights:
        merger.norm.bias = weights["visual.merger.norm.bias"]
        loaded_count += 1
    if "visual.merger.linear_fc1.weight" in weights:
        merger.linear_fc1.weight = weights["visual.merger.linear_fc1.weight"]
        loaded_count += 1
    if "visual.merger.linear_fc1.bias" in weights:
        merger.linear_fc1.bias = weights["visual.merger.linear_fc1.bias"]
        loaded_count += 1
    if "visual.merger.linear_fc2.weight" in weights:
        merger.linear_fc2.weight = weights["visual.merger.linear_fc2.weight"]
        loaded_count += 1
    if "visual.merger.linear_fc2.bias" in weights:
        merger.linear_fc2.bias = weights["visual.merger.linear_fc2.bias"]
        loaded_count += 1

    logger.info(f"Loaded {loaded_count} vision weights")

    # Clean up stored weights
    encoder._vision_weights = None
