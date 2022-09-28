# Long Range Arena Linen
# Copyright (C) 2022  Jason Brown
# 
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
# 
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
# 
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

"""Synthesizer models."""
from functools import partial

from lra_benchmarks.models.synthesizer.synthesizer_attention import SynthesizerSelfAttention
from lra_benchmarks.models.generic.module_collection import ModuleCollection


def get_modules(ignore_dot_product: bool=False, k: int=32,
                synthesizer_mode: str="factorized_random") -> ModuleCollection:
    return ModuleCollection(partial(SynthesizerSelfAttention, ignore_dot_product=ignore_dot_product,
                                    synthesizer_mode=synthesizer_mode, k=k))

