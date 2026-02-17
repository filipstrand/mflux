import mlx.core as mx
from mlx import nn


class CompressedMLP(nn.Module):
    """Two-layer MLP that compresses input to LoRA weight dimensions."""

    def __init__(self, in_dim: int, mid_dim: int, out_dim: int):
        super().__init__()
        self.proj_in = nn.Linear(in_dim, mid_dim, bias=False)
        self.proj_out = nn.Linear(mid_dim, out_dim, bias=False)

    def __call__(self, x: mx.array, residual: mx.array | None = None) -> mx.array:
        x = self.proj_in(x)
        if residual is not None:
            x = x + residual
        x = self.proj_out(x)
        return x


class ImageEmbeddingToLoRAMatrix(nn.Module):
    """Maps an image embedding vector to a pair of LoRA A/B matrices.

    Given embedding x of dim `in_dim`, produces:
      - lora_a: (rank, lora_a_dim)
      - lora_b: (lora_b_dim, rank)
    """

    def __init__(self, in_dim: int, compress_dim: int, lora_a_dim: int, lora_b_dim: int, rank: int):
        super().__init__()
        self.proj_a = CompressedMLP(in_dim, compress_dim, lora_a_dim * rank)
        self.proj_b = CompressedMLP(in_dim, compress_dim, lora_b_dim * rank)
        self.lora_a_dim = lora_a_dim
        self.lora_b_dim = lora_b_dim
        self.rank = rank

    def __call__(self, x: mx.array, residual: mx.array | None = None) -> tuple[mx.array, mx.array]:
        lora_a = self.proj_a(x, residual).reshape(self.rank, self.lora_a_dim)
        lora_b = self.proj_b(x, residual).reshape(self.lora_b_dim, self.rank)
        return lora_a, lora_b


class LoRATrainerBlock(nn.Module):
    """Generates LoRA weights for a single transformer block.

    For a given block_id, produces LoRA A/B pairs for each target layer
    (e.g., attention Q/K/V/O and feed_forward W1/W2/W3).
    """

    def __init__(
        self,
        lora_patterns: list[tuple[str, int, int]],
        prefix: str,
        in_dim: int = 5632,
        compress_dim: int = 128,
        rank: int = 4,
        block_id: int = 0,
    ):
        super().__init__()
        self.prefix = prefix
        self.lora_patterns = lora_patterns
        self.block_id = block_id
        self.layers = [
            ImageEmbeddingToLoRAMatrix(in_dim, compress_dim, lora_a_dim, lora_b_dim, rank)
            for _, lora_a_dim, lora_b_dim in lora_patterns
        ]

    def __call__(self, x: mx.array) -> dict[str, mx.array]:
        lora = {}
        for pattern, layer in zip(self.lora_patterns, self.layers):
            name = pattern[0]
            lora_a, lora_b = layer(x)
            lora[f"{self.prefix}.{self.block_id}.{name}.lora_A.weight"] = lora_a
            lora[f"{self.prefix}.{self.block_id}.{name}.lora_B.weight"] = lora_b
        return lora


class ZImageI2LComponent(nn.Module):
    """Generates LoRA weights for all blocks within a component (layers/context_refiner/noise_refiner)."""

    def __init__(
        self,
        lora_patterns: list[list[tuple[str, int, int]]],
        prefix: str,
        num_blocks: int,
        compress_dim: int = 128,
        rank: int = 4,
    ):
        super().__init__()
        self.blocks = []
        for pattern_group in lora_patterns:
            for block_id in range(num_blocks):
                self.blocks.append(
                    LoRATrainerBlock(
                        lora_patterns=pattern_group,
                        prefix=prefix,
                        compress_dim=compress_dim,
                        rank=rank,
                        block_id=block_id,
                    )
                )

    def __call__(self, x: mx.array) -> dict[str, mx.array]:
        lora = {}
        for block in self.blocks:
            lora.update(block(x))
        return lora


class ZImageI2LDecoder(nn.Module):
    """Z-Image Image-to-LoRA decoder model.

    Takes concatenated image embeddings (SigLIP2 + DINOv3 = 1536 + 4096 = 5632 dim)
    and produces LoRA A/B weight pairs for all target layers of the Z-Image transformer.

    Config from published checkpoint: compress_dim=128, use_residual=False, rank=4.

    Target layers per transformer block:
      - attention: to_q, to_k, to_v, to_out.0 (all 3840×3840)
      - feed_forward: w1 (3840→10240), w2 (10240→3840), w3 (3840→10240)

    Components:
      - layers: 30 transformer blocks
      - context_refiner: 2 blocks
      - noise_refiner: 2 blocks
    """

    def __init__(self, compress_dim: int = 128, rank: int = 4):
        super().__init__()

        lora_patterns = [
            [
                ("attention.to_q", 3840, 3840),
                ("attention.to_k", 3840, 3840),
                ("attention.to_v", 3840, 3840),
                ("attention.to_out.0", 3840, 3840),
            ],
            [
                ("feed_forward.w1", 3840, 10240),
                ("feed_forward.w2", 10240, 3840),
                ("feed_forward.w3", 3840, 10240),
            ],
        ]

        self.layers_lora = ZImageI2LComponent(
            lora_patterns=lora_patterns,
            prefix="layers",
            num_blocks=30,
            compress_dim=compress_dim,
            rank=rank,
        )
        self.context_refiner_lora = ZImageI2LComponent(
            lora_patterns=lora_patterns,
            prefix="context_refiner",
            num_blocks=2,
            compress_dim=compress_dim,
            rank=rank,
        )
        self.noise_refiner_lora = ZImageI2LComponent(
            lora_patterns=lora_patterns,
            prefix="noise_refiner",
            num_blocks=2,
            compress_dim=compress_dim,
            rank=rank,
        )

    def __call__(self, x: mx.array) -> dict[str, mx.array]:
        """
        Args:
            x: (5632,) concatenated SigLIP2 + DINOv3 embedding for a single image.

        Returns:
            Dictionary mapping LoRA weight names to mx.array tensors.
        """
        lora = {}
        lora.update(self.layers_lora(x))
        lora.update(self.context_refiner_lora(x))
        lora.update(self.noise_refiner_lora(x))
        return lora
