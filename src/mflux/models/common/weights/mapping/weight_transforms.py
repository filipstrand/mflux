import mlx.core as mx


class WeightTransforms:
    @staticmethod
    def reshape_gamma_to_1d(tensor: mx.array) -> mx.array:
        if len(tensor.shape) > 1:
            return mx.reshape(tensor, (tensor.shape[0],))
        return tensor

    @staticmethod
    def transpose_patch_embed(tensor: mx.array) -> mx.array:
        if len(tensor.shape) == 5:
            return tensor.transpose(0, 2, 3, 4, 1)
        return tensor

    @staticmethod
    def transpose_conv3d_weight(tensor: mx.array) -> mx.array:
        if len(tensor.shape) == 5:
            return tensor.transpose(0, 2, 3, 4, 1)
        return tensor

    @staticmethod
    def transpose_conv2d_weight(tensor: mx.array) -> mx.array:
        if len(tensor.shape) == 4:
            return tensor.transpose(0, 2, 3, 1)
        return tensor

    @staticmethod
    def transpose_conv_transpose2d_weight(tensor: mx.array) -> mx.array:
        if len(tensor.shape) == 4:
            return tensor.transpose(1, 2, 3, 0)
        return tensor
