"""Reformer language models."""
from functools import partial

from lra_benchmarks.models.reformer import reformer_attention
from lra_benchmarks.models.generic.module_collection import ModuleCollection


def get_modules(chunk_len: int=10, n_chunks_before: int=1, n_hashes: int=1,
                n_buckets: int=10) -> ModuleCollection:
    self_attention = partial(reformer_attention.ReformerSelfAttention,
                             chunk_len=chunk_len, n_chunks_before=n_chunks_before,
                             n_hashes=n_hashes, n_buckets=n_buckets)
    return ModuleCollection(self_attention)

