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

"""Transformer using BigBird (https://arxiv.org/abs/2007.14062)."""

from functools import partial

from lra_benchmarks.models.bigbird.bigbird_attention import BigBirdSelfAttention
from lra_benchmarks.models.generic.module_collection import ModuleCollection


def get_modules(block_size: int=64) -> ModuleCollection:
    return ModuleCollection(partial(BigBirdSelfAttention, block_size=block_size))

