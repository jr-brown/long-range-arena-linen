"""Transformer using BigBird (https://arxiv.org/abs/2007.14062)."""
from functools import partial

from lra_benchmarks.models.bigbird.bigbird_attention import BigBirdSelfAttention
from lra_benchmarks.models.generic.module_collection import ModuleCollection


def get_modules(block_size: int=64) -> ModuleCollection:
    return ModuleCollection(partial(BigBirdSelfAttention, block_size=block_size))

