"""Linformer models."""
from lra_benchmarks.models.linformer.linformer_attention import LinformerSelfAttention
from lra_benchmarks.models.generic.module_collection import ModuleCollection

def get_modules() -> ModuleCollection:
    return ModuleCollection(LinformerSelfAttention)

