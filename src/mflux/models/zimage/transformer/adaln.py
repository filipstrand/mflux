import mlx.core as mx
import mlx.nn as nn


class AdaLayerNorm(nn.Module):
    """Adaptive Layer Normalization with timestep conditioning.

    Produces 6 modulation parameters from timestep embedding:
    - shift_msa, scale_msa, gate_msa (for attention)
    - shift_mlp, scale_mlp, gate_mlp (for FFN)
    """

    DIM = 3840
    NORM_EPS = 1e-5

    def __init__(self):
        super().__init__()
        # Project timestep embedding to 6 modulation parameters
        self.linear = nn.Linear(self.DIM, self.DIM * 6, bias=True)

    def __call__(self, x: mx.array, temb: mx.array) -> tuple[mx.array, ...]:
        """Get modulation parameters from timestep embedding.

        Args:
            x: Input tensor [B, S, dim] (unused, for shape reference)
            temb: Timestep embedding [B, dim]

        Returns:
            Tuple of 6 modulation tensors:
            (shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp)
            Each has shape [B, 1, dim]
        """
        # Get modulation parameters from timestep
        mods = self.linear(nn.silu(temb))
        mods = mods.reshape(temb.shape[0], 1, 6, self.DIM)

        shift_msa = mods[:, :, 0, :]
        scale_msa = mods[:, :, 1, :]
        gate_msa = mods[:, :, 2, :]
        shift_mlp = mods[:, :, 3, :]
        scale_mlp = mods[:, :, 4, :]
        gate_mlp = mods[:, :, 5, :]

        return shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp
