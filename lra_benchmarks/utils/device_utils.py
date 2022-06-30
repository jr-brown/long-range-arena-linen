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

