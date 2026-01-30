import mlx.core as mx
from mlx import nn

from mflux.models.common.config import ModelConfig


class ConvNormOut(nn.Module):
    def __init__(self, num_channels: int = 512, num_groups: int = 32):
        super().__init__()
        self.norm = nn.GroupNorm(
            num_groups=num_groups, dims=num_channels, eps=1e-6, affine=True, pytorch_compatible=True
        )
        # Cache validated output dtype at init time to avoid repeated checks
        # ModelConfig.precision should be an mx.Dtype (has 'size' attribute)
        precision = getattr(ModelConfig, "precision", None)
        self._output_dtype = precision if precision is not None and hasattr(precision, "size") else None

    def __call__(self, input_array: mx.array) -> mx.array:
        input_array = mx.transpose(input_array, (0, 2, 3, 1))
        # Cast to float32 for numerical stability in normalization (matches decoder behavior)
        output_dtype = self._output_dtype if self._output_dtype is not None else input_array.dtype
        output = self.norm(input_array.astype(mx.float32)).astype(output_dtype)
        return mx.transpose(output, (0, 3, 1, 2))
