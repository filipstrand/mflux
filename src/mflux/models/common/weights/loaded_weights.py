from dataclasses import dataclass


@dataclass
class MetaData:
    quantization_level: int | None = None
    mflux_version: str | None = None


@dataclass
class LoadedWeights:
    components: dict[str, dict]
    meta_data: MetaData

    def __getattr__(self, name: str) -> dict | None:
        if name in ("components", "meta_data"):
            return object.__getattribute__(self, name)
        return self.components.get(name)

    def num_transformer_blocks(self) -> int:
        transformer = self.components.get("transformer")
        if transformer and "transformer_blocks" in transformer:
            return len(transformer["transformer_blocks"])
        return 0

    def num_single_transformer_blocks(self) -> int:
        transformer = self.components.get("transformer")
        if transformer and "single_transformer_blocks" in transformer:
            return len(transformer["single_transformer_blocks"])
        return 0
