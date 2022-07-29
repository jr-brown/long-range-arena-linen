"""Local Attention Transformer models."""
from functools import partial

from lra_benchmarks.models.local.local_attention import LocalSelfAttention
from lra_benchmarks.models.generic.module_collection import ModuleCollection


def get_modules(block_size: int=50) -> ModuleCollection:
    return ModuleCollection(partial(LocalSelfAttention, block_size=block_size))

