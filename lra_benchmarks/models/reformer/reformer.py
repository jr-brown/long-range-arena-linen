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

"""Reformer language models."""

from functools import partial

from lra_benchmarks.models.reformer.reformer_attention import ReformerSelfAttention
from lra_benchmarks.models.generic.module_collection import ModuleCollection

def get_modules(chunk_len: int=10, n_chunks_before: int=1, n_hashes: int=1, n_buckets: int=10
                ) -> ModuleCollection:
    return ModuleCollection(partial(ReformerSelfAttention, chunk_len=chunk_len,
                                    n_chunks_before=n_chunks_before, n_hashes=n_hashes,
                                    n_buckets=n_buckets))

