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
"""Local Attention Transformer models."""
from functools import partial

from lra_benchmarks.models.local import local_attention
from lra_benchmarks.models.generic import generic, attention


BLOCK_SIZE = 50
p_pad_length_fn = partial(local_attention.pad_length_fn, block_size=BLOCK_SIZE)
p_attn_fn = partial(local_attention.attention_fn, block_size=BLOCK_SIZE)


LocalSelfAttention = partial(attention.GenericSelfAttention,
                             attention_fn=p_attn_fn,
                             padded_length_fn=p_pad_length_fn)

LocalTransformerBlock = partial(generic.GenericBlock,
                                attention_module=LocalSelfAttention)

LocalTransformerEncoder = partial(generic.GenericEncoder,
                                  block_module=LocalTransformerBlock)

LocalTransformerDualEncoder = partial(generic.GenericDualEncoder,
                                      encoder_module=LocalTransformerEncoder)

LocalTransformerDecoder = partial(generic.GenericDecoder,
                                  block_module=LocalTransformerBlock)

