import logging
import os
import urllib.error
import urllib.request
from pathlib import Path

import mlx.core as mx
import torch
from mlx.utils import tree_unflatten

from mflux.weights.weight_handler import MetaData
from mflux.weights.weight_util import WeightUtil


class WeightHandlerDepthPro:
    def __init__(self, weights: dict, meta_data: MetaData):
        self.weights = weights
        self.meta_data = meta_data

    @staticmethod
    def load_weights() -> "WeightHandlerDepthPro":
        model_path = WeightHandlerDepthPro._download_or_get_cached_weights()
        pt_weights = torch.load(model_path, map_location="cpu")
        weights = WeightHandlerDepthPro._to_mlx_weights(pt_weights)
        weights = [WeightUtil.reshape_weights(k, v) for k, v in weights.items()]
        weights = WeightUtil.flatten(weights)
        weights = tree_unflatten(weights)
        mx.eval(weights)
        return WeightHandlerDepthPro(
            weights=weights,
            meta_data=MetaData(quantization_level=None)
        )  # fmt:off

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
        APPLE_MODEL_URL = "https://ml-site.cdn-apple.com/models/depth-pro/depth_pro.pt"

        # 1. Create cache directory for the model
        cache_dir = Path(os.path.expanduser("~/.cache/mflux/depth_pro"))
        cache_dir.mkdir(parents=True, exist_ok=True)
        model_path = cache_dir / "depth_pro.pt"

        # 2. Download if model doesn't exist
        if not model_path.exists():
            logging.info("Downloading Depth Pro model from Apple...")
            try:
                urllib.request.urlretrieve(APPLE_MODEL_URL, model_path)
                logging.info(f"Downloaded model to {model_path}")
            except (urllib.error.URLError, urllib.error.HTTPError) as e:
                logging.error(f"Failed to download model: {e}")
                logging.info(f"Please manually download from: {APPLE_MODEL_URL}")
                if not model_path.exists():
                    raise FileNotFoundError(f"Model file not found at {model_path}")

        return model_path

    @staticmethod
    def modify_encoder_weights(depth_pro_weights, name):
        tmp = depth_pro_weights.weights["encoder"][name]
        depth_pro_weights.weights["encoder"][name] = {}
        depth_pro_weights.weights["encoder"][name]["layers"] = tmp
