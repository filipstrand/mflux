from dataclasses import dataclass
from typing import List, Protocol


@dataclass
class LoRATarget:
    model_path: str
    possible_up_patterns: List[str]
    possible_down_patterns: List[str]
    possible_alpha_patterns: List[str]


class LoRAMapping(Protocol):
    @staticmethod
    def get_mapping() -> List[LoRATarget]:
        return
