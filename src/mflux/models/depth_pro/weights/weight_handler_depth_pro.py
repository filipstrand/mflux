import logging
import urllib.error
import urllib.request
from dataclasses import dataclass

import mlx.core as mx
import torch

from mflux.cli.defaults.defaults import MFLUX_CACHE_DIR
from mflux.models.common.weights.mapping.weight_mapper import WeightMapper
from mflux.models.depth_pro.weights.depth_pro_weight_mapping import DepthProWeightMapping

logger = logging.getLogger(__name__)

APPLE_MODEL_URL = "https://ml-site.cdn-apple.com/models/depth-pro/depth_pro.pt"


@dataclass
class MetaData:
    quantization_level: int | None


class WeightHandlerDepthPro:
    def __init__(self, weights: dict, meta_data: MetaData):
        self.weights = weights
        self.meta_data = meta_data

    @staticmethod
    def load_weights() -> "WeightHandlerDepthPro":
        # 1. Download or get cached weights
        model_path = WeightHandlerDepthPro._download_or_get_cached_weights()

        # 2. Load PyTorch weights and convert to MLX arrays
        pt_weights = torch.load(model_path, map_location="cpu")
        raw_weights = WeightHandlerDepthPro._to_mlx_weights(pt_weights)

        # 3. Apply declarative weight mapping
        mapping = DepthProWeightMapping.get_mapping()
        mapped_weights = WeightMapper.apply_mapping(raw_weights, mapping)

        return WeightHandlerDepthPro(
            weights=mapped_weights,
            meta_data=MetaData(quantization_level=None),
        )

    @staticmethod
    def _to_mlx_weights(pt_weights) -> dict:
        mlx_weights = {}
        for key, value in pt_weights.items():
            if isinstance(value, torch.Tensor):
                mlx_weights[key] = mx.array(value.numpy())
            else:
                mlx_weights[key] = value
        return mlx_weights

    @staticmethod
    def _download_or_get_cached_weights():
        # 1. Create cache directory for the model
        cache_dir = MFLUX_CACHE_DIR / "depth_pro"

        cache_dir.mkdir(parents=True, exist_ok=True)
        model_path = cache_dir / "depth_pro.pt"

        # 2. Download if model doesn't exist
        if not model_path.exists():
            logger.info("Downloading Depth Pro model from Apple...")
            try:
                urllib.request.urlretrieve(APPLE_MODEL_URL, model_path)
                logger.info(f"Downloaded model to {model_path}")
            except (urllib.error.URLError, urllib.error.HTTPError) as e:
                logger.error(f"Failed to download model: {e}")
                logger.info(f"Please manually download from: {APPLE_MODEL_URL}")
                if not model_path.exists():
                    raise FileNotFoundError(f"Model file not found at {model_path}")

        return model_path
