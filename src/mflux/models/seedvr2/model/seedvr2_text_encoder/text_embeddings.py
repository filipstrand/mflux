from pathlib import Path

import mlx.core as mx

_EMBEDDINGS_DIR = Path(__file__).parent / "embeddings"
_POS_EMB_PATH = _EMBEDDINGS_DIR / "pos_emb.safetensors"


class SeedVR2TextEmbeddings:
    POS_EMB_SHAPE = (58, 5120)

    @staticmethod
    def load_positive(batch_size: int = 1) -> mx.array:
        emb = mx.load(str(_POS_EMB_PATH))["embedding"]
        if emb.ndim == 2:
            emb = emb[None, ...]

        if batch_size > 1:
            emb = mx.repeat(emb, batch_size, axis=0)

        return emb
