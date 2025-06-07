from mflux import ModelConfig
from mflux.controlnet.transformer_controlnet import TransformerControlnet
from mflux.controlnet.weight_handler_controlnet import WeightHandlerControlnet
from mflux.flux_tools.redux.weight_handler_redux import WeightHandlerRedux
from mflux.models.depth_pro.depth_pro import DepthPro
from mflux.models.redux_encoder.redux_encoder import ReduxEncoder
from mflux.models.siglip_vision_transformer.siglip_vision_transformer import SiglipVisionTransformer
from mflux.models.text_encoder.clip_encoder.clip_encoder import CLIPEncoder
from mflux.models.text_encoder.t5_encoder.t5_encoder import T5Encoder
from mflux.models.transformer.transformer import Transformer
from mflux.models.vae.vae import VAE
from mflux.tokenizer.clip_tokenizer import TokenizerCLIP
from mflux.tokenizer.t5_tokenizer import TokenizerT5
from mflux.tokenizer.tokenizer_handler import TokenizerHandler
from mflux.weights.weight_handler import WeightHandler
from mflux.weights.weight_handler_lora import WeightHandlerLoRA
from mflux.weights.weight_handler_lora_huggingface import WeightHandlerLoRAHuggingFace
from mflux.weights.weight_util import WeightUtil


class FluxInitializer:
    @staticmethod
    def init(
        flux_model,
        model_config: ModelConfig,
        quantize: int | None,
        local_path: str | None,
        lora_paths: list[str] | None = None,
        lora_scales: list[float] | None = None,
        lora_names: list[str] | None = None,
        lora_repo_id: str | None = None,
        custom_transformer=None,
    ) -> None:
        # 0. Set paths, configs, and prompt_cache for later
        lora_paths = lora_paths or []
        flux_model.prompt_cache = {}
        flux_model.model_config = model_config

        # 1. Initialize tokenizers
        tokenizers = TokenizerHandler(
            repo_id=model_config.model_name,
            max_t5_length=model_config.max_sequence_length,
            local_path=local_path,
        )
        flux_model.t5_tokenizer = TokenizerT5(
            tokenizer=tokenizers.t5,
            max_length=model_config.max_sequence_length,
        )
        flux_model.clip_tokenizer = TokenizerCLIP(
            tokenizer=tokenizers.clip,
        )

        # 2. Load the regular weights
        weights = WeightHandler.load_regular_weights(
            repo_id=model_config.model_name,
            local_path=local_path,
            transformer_repo_id=model_config.custom_transformer_model,
        )

        # 3. Initialize all models
        flux_model.vae = VAE()
        flux_model.t5_text_encoder = T5Encoder()
        flux_model.clip_text_encoder = CLIPEncoder()
        if custom_transformer is not None:
            flux_model.transformer = custom_transformer
        else:
            flux_model.transformer = Transformer(
                model_config=model_config,
                num_transformer_blocks=weights.num_transformer_blocks(),
                num_single_transformer_blocks=weights.num_single_transformer_blocks(),
            )

        # 4. Apply weights and quantize the models
        flux_model.bits = WeightUtil.set_weights_and_quantize(
            quantize_arg=quantize,
            weights=weights,
            vae=flux_model.vae,
            transformer=flux_model.transformer,
            t5_text_encoder=flux_model.t5_text_encoder,
            clip_text_encoder=flux_model.clip_text_encoder,
        )

        # 5. Set LoRA weights
        hf_lora_paths = WeightHandlerLoRAHuggingFace.download_loras(
            lora_names=lora_names,
            repo_id=lora_repo_id,
        )
        flux_model.lora_paths = lora_paths + hf_lora_paths
        flux_model.lora_scales = (lora_scales or []) + [1.0] * len(hf_lora_paths)
        lora_weights = WeightHandlerLoRA.load_lora_weights(
            transformer=flux_model.transformer,
            lora_files=flux_model.lora_paths,
            lora_scales=flux_model.lora_scales,
        )
        WeightHandlerLoRA.set_lora_weights(
            transformer=flux_model.transformer,
            loras=lora_weights,
        )

    @staticmethod
    def init_depth(
        flux_model,
        model_config: ModelConfig,
        quantize: int | None,
        local_path: str | None,
        lora_paths: list[str] | None = None,
        lora_scales: list[float] | None = None,
        lora_names: list[str] | None = None,
        lora_repo_id: str | None = None,
    ):
        # 1. Start with the same init as regular Flux
        FluxInitializer.init(
            flux_model=flux_model,
            model_config=model_config,
            quantize=quantize,
            local_path=local_path,
            lora_paths=lora_paths,
            lora_scales=lora_scales,
            lora_names=lora_names,
            lora_repo_id=lora_repo_id,
        )

        # 2. Initialize the DepthPro model
        flux_model.depth_pro = DepthPro()

    @staticmethod
    def init_redux(
        flux_model,
        quantize: int | None,
        local_path: str | None,
        lora_paths: list[str] | None = None,
        lora_scales: list[float] | None = None,
        lora_names: list[str] | None = None,
        lora_repo_id: str | None = None,
    ):
        # 1. Start with the same init as regular Flux dev
        FluxInitializer.init(
            flux_model=flux_model,
            model_config=ModelConfig.dev(),
            quantize=quantize,
            local_path=local_path,
            lora_paths=lora_paths,
            lora_scales=lora_scales,
            lora_names=lora_names,
            lora_repo_id=lora_repo_id,
        )

        # 2. Initialize the redux specific addons
        redux_weights = WeightHandlerRedux.load_weights()
        flux_model.image_embedder = ReduxEncoder()
        flux_model.image_encoder = SiglipVisionTransformer()
        WeightUtil.set_redux_weights_and_quantize(
            quantize_arg=quantize,
            weights=redux_weights,
            redux_encoder=flux_model.image_embedder,
            siglip_vision_transformer=flux_model.image_encoder,
        )

    @staticmethod
    def init_controlnet(
        flux_model,
        model_config: ModelConfig,
        quantize: int | None,
        local_path: str | None,
        lora_paths: list[str] | None = None,
        lora_scales: list[float] | None = None,
        lora_names: list[str] | None = None,
        lora_repo_id: str | None = None,
    ) -> None:
        # 1. Start with the same init as regular Flux
        FluxInitializer.init(
            flux_model=flux_model,
            model_config=model_config,
            quantize=quantize,
            local_path=local_path,
            lora_paths=lora_paths,
            lora_scales=lora_scales,
            lora_names=lora_names,
            lora_repo_id=lora_repo_id,
        )

        # 2. Apply ControlNet-specific initialization
        weights_controlnet = WeightHandlerControlnet.load_controlnet_transformer(
            controlnet_model=model_config.controlnet_model
        )
        flux_model.transformer_controlnet = TransformerControlnet(
            model_config=model_config,
            num_transformer_blocks=weights_controlnet.num_transformer_blocks(),
            num_single_transformer_blocks=weights_controlnet.num_single_transformer_blocks(),
        )
        WeightUtil.set_controlnet_weights_and_quantize(
            quantize_arg=quantize,
            weights=weights_controlnet,
            transformer_controlnet=flux_model.transformer_controlnet,
        )

    @staticmethod
    def init_concept(
        flux_model,
        model_config: ModelConfig,
        quantize: int | None,
        local_path: str | None,
        lora_paths: list[str] | None = None,
        lora_scales: list[float] | None = None,
        lora_names: list[str] | None = None,
        lora_repo_id: str | None = None,
    ):
        # Import here to avoid circular dependency
        from mflux.community.concept_attention.transformer_concept import TransformerConcept

        # 1. Load weights first to get transformer dimensions
        weights = WeightHandler.load_regular_weights(
            repo_id=model_config.model_name,
            local_path=local_path,
        )

        # 2. Create custom TransformerConcept
        custom_transformer = TransformerConcept(
            model_config=model_config,
            num_transformer_blocks=weights.num_transformer_blocks(),
            num_single_transformer_blocks=weights.num_single_transformer_blocks(),
        )

        # 3. Use the improved FluxInitializer with custom transformer
        FluxInitializer.init(
            flux_model=flux_model,
            model_config=model_config,
            quantize=quantize,
            local_path=local_path,
            lora_paths=lora_paths,
            lora_scales=lora_scales,
            lora_names=lora_names,
            lora_repo_id=lora_repo_id,
            custom_transformer=custom_transformer,
        )
