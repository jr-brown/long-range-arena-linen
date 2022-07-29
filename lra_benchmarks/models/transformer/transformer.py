"""Transformer model."""
from lra_benchmarks.models.generic.module_collection import ModuleCollection
from lra_benchmarks.models.generic import attention

def get_modules() -> ModuleCollection:
    return ModuleCollection(attention.MaskedSelfAttention)

