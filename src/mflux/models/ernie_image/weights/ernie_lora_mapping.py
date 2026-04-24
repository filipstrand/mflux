from mflux.models.common.lora.mapping.lora_mapping import LoRAMapping, LoRATarget


class ErnieLoRAMapping(LoRAMapping):
    @staticmethod
    def get_mapping() -> list[LoRATarget]:
        return []
