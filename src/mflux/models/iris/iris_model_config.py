from mflux.models.common.config.model_config import ModelConfig


class IrisModelConfig:
    REPO = "speridlabs/iris-3b"

    @staticmethod
    def turbo() -> ModelConfig:
        return IrisModelConfig._build(["iris-3b-turbo", "iris-turbo"], supports_guidance=False)

    @staticmethod
    def base() -> ModelConfig:
        return IrisModelConfig._build(["iris-3b", "iris"], supports_guidance=True)

    @staticmethod
    def _build(aliases: list[str], supports_guidance: bool) -> ModelConfig:
        return ModelConfig(
            priority=12,
            aliases=aliases,
            model_name=IrisModelConfig.REPO,
            base_model=None,
            controlnet_model=None,
            custom_transformer_model=None,
            num_train_steps=1000,
            max_sequence_length=300,
            supports_guidance=supports_guidance,
            requires_sigma_shift=False,
            text_encoder_overrides={
                "hidden_size": 2560,
                "intermediate_size": 9728,
                "num_hidden_layers": 36,
                "rope_theta": 5000000.0,
            },
        )
