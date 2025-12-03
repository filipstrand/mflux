import mlx.core as mx


class LoraTransforms:
    @staticmethod
    def split_q_up(tensor: mx.array) -> mx.array:
        return LoraTransforms._split_qkv_up(tensor, 0)

    @staticmethod
    def split_k_up(tensor: mx.array) -> mx.array:
        return LoraTransforms._split_qkv_up(tensor, 1)

    @staticmethod
    def split_v_up(tensor: mx.array) -> mx.array:
        return LoraTransforms._split_qkv_up(tensor, 2)

    @staticmethod
    def split_q_down(tensor: mx.array) -> mx.array:
        return LoraTransforms._split_qkv_down(tensor, 0)

    @staticmethod
    def split_k_down(tensor: mx.array) -> mx.array:
        return LoraTransforms._split_qkv_down(tensor, 1)

    @staticmethod
    def split_v_down(tensor: mx.array) -> mx.array:
        return LoraTransforms._split_qkv_down(tensor, 2)

    @staticmethod
    def split_single_q_up(tensor: mx.array) -> mx.array:
        return LoraTransforms._split_qkv_mlp_up(tensor, 0)

    @staticmethod
    def split_single_k_up(tensor: mx.array) -> mx.array:
        return LoraTransforms._split_qkv_mlp_up(tensor, 1)

    @staticmethod
    def split_single_v_up(tensor: mx.array) -> mx.array:
        return LoraTransforms._split_qkv_mlp_up(tensor, 2)

    @staticmethod
    def split_single_mlp_up(tensor: mx.array) -> mx.array:
        return LoraTransforms._split_qkv_mlp_up(tensor, 3)

    @staticmethod
    def split_single_q_down(tensor: mx.array) -> mx.array:
        return LoraTransforms._split_qkv_mlp_down(tensor, 0)

    @staticmethod
    def split_single_k_down(tensor: mx.array) -> mx.array:
        return LoraTransforms._split_qkv_mlp_down(tensor, 1)

    @staticmethod
    def split_single_v_down(tensor: mx.array) -> mx.array:
        return LoraTransforms._split_qkv_mlp_down(tensor, 2)

    @staticmethod
    def split_single_mlp_down(tensor: mx.array) -> mx.array:
        return LoraTransforms._split_qkv_mlp_down(tensor, 3)

    @staticmethod
    def _transpose(tensor: mx.array) -> mx.array:
        return tensor.T

    @staticmethod
    def _split_qkv_up(tensor: mx.array, index: int, num_splits: int = 3) -> mx.array:
        split_size = tensor.shape[0] // num_splits
        start = index * split_size
        end = start + split_size
        return tensor[start:end, :]

    @staticmethod
    def _split_qkv_down(tensor: mx.array, index: int, num_splits: int = 3) -> mx.array:
        rank = tensor.shape[0]
        if rank % num_splits == 0:
            chunk_size = rank // num_splits
            start = index * chunk_size
            end = start + chunk_size
            return tensor[start:end, :]
        else:
            return tensor

    @staticmethod
    def _split_qkv_mlp_up(tensor: mx.array, index: int, dims: list[int] | None = None) -> mx.array:
        if dims is None:
            dims = [3072, 3072, 3072, 12288]

        start = sum(dims[:index])
        end = start + dims[index]
        return tensor[start:end, :]

    @staticmethod
    def _split_qkv_mlp_down(tensor: mx.array, index: int, num_splits: int = 4) -> mx.array:
        rank = tensor.shape[0]
        if rank % num_splits == 0:
            chunk_size = rank // num_splits
            start = index * chunk_size
            end = start + chunk_size
            return tensor[start:end, :]
        else:
            return tensor
