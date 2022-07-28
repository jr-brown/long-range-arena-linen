"""Local Attention Transformer models."""
from functools import partial

from lra_benchmarks.models.local import local_attention
from lra_benchmarks.models.generic import attention
from lra_benchmarks.models.generic.module_collection import ModuleCollection


def get_modules(block_size: int=50) -> ModuleCollection:
    p_pad_length_fn = partial(attention.pad_length_fn, block_size=block_size)
    p_attn_fn = partial(local_attention.attention_fn, block_size=block_size)

    LocalSelfAttention = partial(attention.GenericSelfAttention,
                                 attention_fn=p_attn_fn,
                                 padded_length_fn=p_pad_length_fn)

    return ModuleCollection(LocalSelfAttention)

