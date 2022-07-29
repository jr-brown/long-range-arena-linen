"""LinearTransformer model."""
from lra_benchmarks.models.linear_transformer.linear_attention import LinearSelfAttention
from lra_benchmarks.models.generic.module_collection import ModuleCollection


def get_modules() -> ModuleCollection:
    return ModuleCollection(LinearSelfAttention)

