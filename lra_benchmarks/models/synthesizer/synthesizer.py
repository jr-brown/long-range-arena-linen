"""Synthesizer models."""
from functools import partial

from lra_benchmarks.models.synthesizer.synthesizer_attention import SynthesizerSelfAttention
from lra_benchmarks.models.generic.module_collection import ModuleCollection


def get_modules(ignore_dot_product: bool=True, k: int=32,
                synthesizer_mode: str="factorized_random") -> ModuleCollection:
    return ModuleCollection(partial(SynthesizerSelfAttention, ignore_dot_product=ignore_dot_product,
                                    synthesizer_mode=synthesizer_mode, k=k))

