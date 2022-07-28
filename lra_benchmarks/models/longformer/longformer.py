# Copyright 2021 Google LLC

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     https://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Longformer modules."""
from functools import partial

from lra_benchmarks.models.longformer import longformer_attention
from lra_benchmarks.models.generic.module_collection import ModuleCollection


def get_modules(sliding_window_size: int=512) -> ModuleCollection:
    LongformerAttention = partial(longformer_attention.LongformerAttention,
                                  sliding_window_size=sliding_window_size)
    return ModuleCollection(LongformerAttention)

