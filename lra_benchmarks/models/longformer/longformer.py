"""Longformer modules."""
from functools import partial

from lra_benchmarks.models.longformer.longformer_attention import LongformerSelfAttention
from lra_benchmarks.models.generic.module_collection import ModuleCollection


def get_modules(sliding_window_size: int=512) -> ModuleCollection:
    # Use block kwargs in the generic high level calls to give global masks
    return ModuleCollection(partial(LongformerSelfAttention,
                                    sliding_window_size=sliding_window_size))

