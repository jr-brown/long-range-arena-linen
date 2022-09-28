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

from typing import Any
import yaml

from lra_benchmarks.utils.misc_utils import recursive_dict_update


def load_many_configs(cfg_paths: list[str]) -> list[dict[str, Any]]:
    cfgs = []
    for path in cfg_paths:
        with open(path, 'r') as f:
            new_cfg = yaml.full_load(f)
            if not isinstance(new_cfg, dict):
                raise ValueError(f"Config from path {path} did not load into a dict")
            cfgs.append(new_cfg)
    return cfgs


def load_many_configs_into_one(cfg_paths: list[str]) -> dict[str, Any]:
    config_dict = {}
    for tgt_cfg in load_many_configs(cfg_paths=cfg_paths):
        config_dict = recursive_dict_update(config_dict, tgt_cfg)
    return config_dict


