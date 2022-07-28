"""Local Attention Transformer models."""
from functools import partial

from lra_benchmarks.models.local import local_attention
from lra_benchmarks.models.generic import attention, generic
from lra_benchmarks.models.generic.module_collection import ModuleCollection


def get_modules(block_size: int=50) -> ModuleCollection:
    p_pad_length_fn = partial(attention.pad_length_fn, block_size=block_size)
    LocalAttention = partial(local_attention.LocalAttention, block_size=block_size)
    block = partial(generic.GenericBlock, attention_module=LocalAttention,
                    padded_length_fn=p_pad_length_fn)

    return ModuleCollection(block)

