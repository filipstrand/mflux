from mflux.models.fibo.weights import FIBOWeightHandler
from mflux.models.fibo_vlm.model.qwen3_vl_decoder import Qwen3VLDecoder
from mflux.models.fibo_vlm.model.qwen3_vl_vision_model import Qwen3VLVisionModel


class FIBOVLMInitializer:
    def __init__(self, config):
        self.config = config

    @staticmethod
    def init_vlm(
        vlm_model,
        model_id: str = "briaai/FIBO-vlm",
        local_path: str | None = None,
    ) -> None:
        # 1. Load VLM weights
        weights = FIBOWeightHandler.load_vlm_regular_weights(
            repo_id=model_id,
            local_path=local_path,
        )

        # 2. Initialize decoder model
        config = weights.config
        vlm_model.decoder = Qwen3VLDecoder(
            vocab_size=config["vocab_size"],
            hidden_size=config["hidden_size"],
            num_hidden_layers=config["num_hidden_layers"],
            num_attention_heads=config["num_attention_heads"],
            num_key_value_heads=config["num_key_value_heads"],
            intermediate_size=config["intermediate_size"],
            max_position_embeddings=config["max_position_embeddings"],
            rope_theta=config["rope_theta"],
            rms_norm_eps=config["rms_norm_eps"],
            head_dim=config["head_dim"],
            attention_bias=config["attention_bias"],
            mrope_section=config["mrope_section"],
            attention_scaling=config.get("attention_scaling", 1.0),
        )

        # 3. Apply weights to decoder
        if weights.decoder:
            vlm_model.decoder.update(weights.decoder, strict=False)

        # 4. Initialize visual encoder (for image input support)
        vision_config = config.get("vision_config")
        if vision_config:
            vlm_model.decoder.visual = Qwen3VLVisionModel(
                patch_size=vision_config["patch_size"],
                temporal_patch_size=vision_config["temporal_patch_size"],
                in_channels=vision_config["in_channels"],
                hidden_size=vision_config["hidden_size"],
                num_heads=vision_config["num_heads"],
                intermediate_size=vision_config["intermediate_size"],
                depth=vision_config["depth"],
                spatial_merge_size=vision_config["spatial_merge_size"],
                num_position_embeddings=vision_config["num_position_embeddings"],
                out_hidden_size=vision_config["out_hidden_size"],
                deepstack_visual_indexes=vision_config["deepstack_visual_indexes"],
                hidden_act=vision_config["hidden_act"],
            )

            # Load visual encoder weights if available
            if hasattr(weights, "visual") and weights.visual:
                vlm_model.decoder.visual.update(weights.visual, strict=False)

        # 5. Set image_token_id from config
        vlm_model.decoder.image_token_id = config.get("image_token_id")

        # 6. Initialize processor for tokenization
        try:
            from transformers import AutoTokenizer

            from mflux.models.fibo.tokenizer.qwen2vl_processor import Qwen2VLProcessor

            tokenizer = AutoTokenizer.from_pretrained(model_id)
            vlm_model.processor = Qwen2VLProcessor(tokenizer=tokenizer)
        except (ImportError, OSError, ValueError) as e:
            raise RuntimeError(f"Failed to load tokenizer from {model_id}: {e}") from e

        # 7. Extract token IDs
        tokenizer = vlm_model.processor.tokenizer
        vlm_model.eos_token_id = tokenizer.eos_token_id
        vlm_model.pad_token_id = tokenizer.pad_token_id or tokenizer.eos_token_id
        if isinstance(vlm_model.eos_token_id, list):
            vlm_model.eos_token_id = vlm_model.eos_token_id[0] if vlm_model.eos_token_id else None

        # 8. Store model ID and local path
        vlm_model.model_id = model_id
        vlm_model.local_path = local_path
