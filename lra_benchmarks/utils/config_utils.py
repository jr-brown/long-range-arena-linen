from typing import Any
from copy import deepcopy
import yaml

from flax.core import FrozenDict


def is_dict(x):
    return isinstance(x, dict) or isinstance(x, FrozenDict)


def recursive_dict_update(base: dict, target: dict, assert_type_match=True,
                          type_match_ignore_nones=True) -> dict:
    new = deepcopy(base)

    for k, v in target.items():

        if assert_type_match and (k in base.keys()):
            t1, t2 = type(base[k]), type(v)
            if t1 != t2 and (not type_match_ignore_nones or (t1 is None or t2 is None)):
                raise ValueError(f"Types do not match for key {k}, {t1} vs {t2}")

        if (k not in base.keys()) or (not is_dict(v)) or (not is_dict(base[k])):
            new[k] = v
        else:
            new[k] = recursive_dict_update(base[k], v)

    return new


def load_configs(cfg_paths: list[str]) -> dict[str, Any]:
    cfgs = []
    for path in cfg_paths:
        with open(path, 'r') as f:
            new_cfg = yaml.full_load(f)
            if not isinstance(new_cfg, dict):
                raise ValueError(f"Config from path {path} did not load into a dict")
            cfgs.append(new_cfg)

    config_dict = {}
    for tgt_cfg in cfgs:
        config_dict = recursive_dict_update(config_dict, tgt_cfg)

    return config_dict

