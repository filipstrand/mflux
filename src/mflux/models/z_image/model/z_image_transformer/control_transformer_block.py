from mlx import nn

from mflux.models.z_image.model.z_image_transformer.transformer_block import ZImageTransformerBlock


class ZImageControlTransformerBlock(ZImageTransformerBlock):
    """A single VACE-style control block (sc-2257).

    Mirrors a base :class:`ZImageTransformerBlock` (identical attention / FFN /
    adaLN submodules and weight keys) and adds the two zero-initialised
    projections that thread the control hidden state through the parallel
    control stack:

      - ``before_proj`` (block 0 only): projects the incoming control context and
        adds the base hidden state once, seeding the control branch.
      - ``after_proj`` (every block): the zero-init projection whose output is the
        per-block *hint* added back into the base transformer at the matching
        place.

    Ported from VideoX-Fun ``videox_fun/models/z_image_transformer2d_control.py``
    (``ZImageControlTransformerBlock``), Apache-2.0. The forward threading is
    implemented at the transformer level (see ``ZImageControlTransformer``) rather
    than via torch ``stack``/``unbind`` so MLX can run it as a plain Python loop.
    """

    def __init__(
        self,
        dim: int,
        n_heads: int,
        norm_eps: float = 1e-5,
        qk_norm: bool = True,
        has_before_proj: bool = False,
    ):
        super().__init__(dim=dim, n_heads=n_heads, norm_eps=norm_eps, qk_norm=qk_norm)
        if has_before_proj:
            self.before_proj = nn.Linear(dim, dim, bias=True)
        self.after_proj = nn.Linear(dim, dim, bias=True)
