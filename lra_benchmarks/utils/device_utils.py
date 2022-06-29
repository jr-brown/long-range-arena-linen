import jax
from typing import Optional

def get_devices(ids: Optional[list[int]]) -> list:
    if ids is not None:
        return [x for x in jax.local_devices() if x.id in ids]
    else:
        return jax.devices()

def shard(xs, n_devices=None):
    n = n_devices if n_devices is not None else jax.local_device_count()
    return jax.tree_map(lambda x: x.reshape((n, -1) + x.shape[1:]), xs)

