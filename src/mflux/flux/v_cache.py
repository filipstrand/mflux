import mlx.core as mx
import numpy as np


class VCache:
    is_inverting = True
    v_cache = {}
    t_max = 5

    @staticmethod
    def save_dict(data_dict, filename):
        np_dict = {k: v.tolist() for k, v in data_dict.items()}
        np.savez_compressed(filename, **np_dict)

    @staticmethod
    def load_dict(filename):
        data = np.load(filename)
        return {k: mx.array(v) for k, v in data.items()}
