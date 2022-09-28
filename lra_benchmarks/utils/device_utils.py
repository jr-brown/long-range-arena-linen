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

import jax
from typing import Optional

def get_devices(ids: Optional[list[int]]) -> tuple[Optional[list], int]:
    if ids is not None:
        gpu_devices = [x for x in jax.local_devices() if x.id in ids]
    else:
        gpu_devices = jax.devices()

    if gpu_devices == []:
        return None, 1
    else:
        return gpu_devices, len(gpu_devices)

def shard(xs, n_devices=None):
    n = n_devices if n_devices is not None else jax.local_device_count()
    return jax.tree_map(lambda x: x.reshape((n, -1) + x.shape[1:]), xs)

