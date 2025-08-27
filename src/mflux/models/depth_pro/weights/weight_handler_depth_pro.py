import logging
import urllib.error
import urllib.request

import mlx.core as mx
import torch
from mlx.utils import tree_unflatten

from mflux.ui.defaults import MFLUX_CACHE_DIR
from mflux.weights.weight_handler import MetaData
from mflux.weights.weight_util import WeightUtil

logger = logging.getLogger(__name__)


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

    @staticmethod
    def reposition_encoder_weights(depth_pro_weights, name):
        tmp = depth_pro_weights.weights["encoder"][name]
        depth_pro_weights.weights["encoder"][name] = {}
        depth_pro_weights.weights["encoder"][name]["layers"] = tmp

    @staticmethod
    def reposition_head_weights(depth_pro_weights):
        tmp = depth_pro_weights.weights["head"]
        depth_pro_weights.weights["head"] = {}
        depth_pro_weights.weights["head"]["convs"] = tmp

    @staticmethod
    def reshape_transposed_convolution_weights(depth_pro_weights):
        WeightHandlerDepthPro._reshape_upsample(depth_pro_weights, "upsample_latent0", 1)
        WeightHandlerDepthPro._reshape_upsample(depth_pro_weights, "upsample_latent0", 2)
        WeightHandlerDepthPro._reshape_upsample(depth_pro_weights, "upsample_latent0", 3)

        WeightHandlerDepthPro._reshape_upsample(depth_pro_weights, "upsample_latent1", 1)
        WeightHandlerDepthPro._reshape_upsample(depth_pro_weights, "upsample_latent1", 2)

        WeightHandlerDepthPro._reshape_upsample(depth_pro_weights, "upsample0", 1)
        WeightHandlerDepthPro._reshape_upsample(depth_pro_weights, "upsample1", 1)
        WeightHandlerDepthPro._reshape_upsample(depth_pro_weights, "upsample2", 1)

        WeightHandlerDepthPro._reshape_upsample_lowres(depth_pro_weights)

        WeightHandlerDepthPro._reshape_deconv(depth_pro_weights, 1)
        WeightHandlerDepthPro._reshape_deconv(depth_pro_weights, 2)
        WeightHandlerDepthPro._reshape_deconv(depth_pro_weights, 3)
        WeightHandlerDepthPro._reshape_deconv(depth_pro_weights, 4)

        WeightHandlerDepthPro._reshape_head(depth_pro_weights, 1)

    @staticmethod
    def _reshape_upsample(depth_pro_weights, name, layer):
        tmp = depth_pro_weights.weights["encoder"][name]["layers"][layer]["weight"]
        tmp = WeightHandlerDepthPro._reshape(tmp)
        depth_pro_weights.weights["encoder"][name]["layers"][layer]["weight"] = tmp

    @staticmethod
    def _reshape_upsample_lowres(depth_pro_weights):
        tmp = depth_pro_weights.weights["encoder"]["upsample_lowres"]["weight"]
        tmp = WeightHandlerDepthPro._reshape(tmp)
        depth_pro_weights.weights["encoder"]["upsample_lowres"]["weight"] = tmp

    @staticmethod
    def _reshape_deconv(depth_pro_weights, layer):
        tmp = depth_pro_weights.weights["decoder"]["fusions"][layer]["deconv"]["weight"]
        tmp = WeightHandlerDepthPro._reshape(tmp)
        depth_pro_weights.weights["decoder"]["fusions"][layer]["deconv"]["weight"] = tmp

    @staticmethod
    def _reshape_head(depth_pro_weights, layer):
        tmp = depth_pro_weights.weights["head"]["convs"][layer]["weight"]
        tmp = WeightHandlerDepthPro._reshape(tmp)
        depth_pro_weights.weights["head"]["convs"][layer]["weight"] = tmp

    @staticmethod
    def _reshape(tensor):
        tensor = tensor.transpose(0, 3, 1, 2)
        tensor = tensor.transpose(1, 0, 2, 3)
        tensor = tensor.transpose(0, 2, 3, 1)
        return tensor
