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
"""Transformer model."""
from functools import partial

from lra_benchmarks.models.generic import generic, attention

TransformerBlock = partial(generic.GenericBlock,
                           attention_module=attention.MaskedSelfAttention)
TransformerEncoder = partial(generic.GenericEncoder, block_module=TransformerBlock)
TransformerDualEncoder = partial(generic.GenericDualEncoder,
                                 encoder_module=TransformerEncoder)

