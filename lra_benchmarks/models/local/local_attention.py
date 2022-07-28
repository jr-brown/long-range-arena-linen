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
"""Local Attention modules."""
from flax import linen as nn
import jax.numpy as jnp

from lra_benchmarks.utils.array_utils import make_block_attention_mask


def attention_fn(query, key, value, *,
                 block_size,
                 causal_mask: bool=False,
                 padding_mask=None,
                 key_padding_mask=None,
                 segmentation=None,
                 key_segmentation=None,
                 dropout_rng=None,
                 dropout_rate: float=0,
                 broadcast_dropout: bool=True,
                 deterministic: bool=False,
                 dtype=None,
                 precision=None):

    # Input dimensions are [batch_size, length, num_heads, head_dim]
    assert query.ndim == 4
    assert query.shape == key.shape == value.shape

    bs = query.shape[0]
    qlength = query.shape[1]
    kvlength = key.shape[1]
    num_heads = query.shape[2]
    head_dim = query.shape[3]

    # block reshape before attention
    num_query_blocks = qlength // block_size
    num_kv_blocks = kvlength // block_size

    block_query = jnp.reshape(query, (bs, num_query_blocks, block_size, num_heads, head_dim))
    block_key = jnp.reshape(key, (bs, num_kv_blocks, block_size, num_heads, head_dim))
    block_value = jnp.reshape(value, (bs, num_kv_blocks, block_size, num_heads, head_dim))

    _, attention_bias = make_block_attention_mask(
        seq_shape=key[:-2],
        bs=bs,
        num_query_blocks=num_query_blocks,
        block_size=block_size,
        num_heads=num_heads,
        dtype=dtype,
        causal_mask=causal_mask,
        padding_mask=padding_mask,
        key_padding_mask=key_padding_mask,
        segmentation=segmentation,
        key_segmentation=key_segmentation,
        use_attention_bias=True,
    )

    x = nn.dot_product_attention(
            block_query,
            block_key,
            block_value,
            dtype=dtype,
            bias=attention_bias,
            precision=precision,
            dropout_rng=dropout_rng,
            dropout_rate=dropout_rate,
            broadcast_dropout=broadcast_dropout,
            deterministic=deterministic)

    return jnp.reshape(x, (bs, qlength, num_heads, head_dim))

