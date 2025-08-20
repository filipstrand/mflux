from typing import TYPE_CHECKING

from ....utils import DIFFUSERS_SLOW_IMPORT, _LazyModule


_import_structure = {"pipeline_stochastic_karras_ve": ["KarrasVePipeline"]}

if TYPE_CHECKING or DIFFUSERS_SLOW_IMPORT:
    from .pipeline_stochastic_karras_ve import KarrasVePipeline

else:
    import sys

    sys.modules[__name__] = _LazyModule(
        __name__,
        globals()["__file__"],
        _import_structure,
        module_spec=__spec__,
    )
