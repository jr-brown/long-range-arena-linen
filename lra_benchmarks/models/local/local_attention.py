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
from typing import Any, Tuple, Optional, Callable
from functools import partial

from flax import linen as nn
from flax.linen.linear import PrecisionLike, default_kernel_init

import jax.numpy as jnp
import jax.nn as jnn

from lra_benchmarks.utils.array_utils import make_block_attention_mask

PRNGKey = Any
Shape = Tuple[int, ...]
Dtype = Any
Array = Any


class LocalAttentionFN(nn.Module):

    num_heads: int
    head_dim: int
    block_size: int
    dtype: Optional[Dtype]=None
    param_dtype: Dtype=jnp.float32
    broadcast_dropout: bool=True
    dropout_rate: float=0
    precision: PrecisionLike=None
    kernel_init: Callable[[PRNGKey, Shape, Dtype], Array]=default_kernel_init
    bias_init: Callable[[PRNGKey, Shape, Dtype], Array]=jnn.initializers.zeros
    use_bias: bool=True
    use_attention_bias: bool=False

    @nn.compact
    def __call__(self, inputs_q, inputs_kv, *, causal_mask: bool=False, padding_mask=None,
                 key_padding_mask=None, segmentation=None, key_segmentation=None,
                 deterministic: bool=False):

        dense = partial(
            nn.DenseGeneral,
            axis=-1,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            features=(self.num_heads, self.head_dim),
            kernel_init=self.kernel_init,
            bias_init=self.bias_init,
            use_bias=self.use_bias,
            precision=self.precision
        )

        # project inputs_q to multi-headed q/k/v
        # dimensions are then [batch, length, n_heads, n_features_per_head]
        qd, kd, vd = dense(name='query'), dense(name='key'), dense(name='value')
        query, key, value = qd(inputs_q), kd(inputs_kv), vd(inputs_kv)

        if self.dropout_rate > 0 and not deterministic:
            dropout_rng = self.make_rng('dropout')
        else:
            dropout_rng = None

        # Input dimensions are [batch_size, length, num_heads, head_dim]
        assert query.ndim == 4
        assert query.shape == key.shape == value.shape

        bs = query.shape[0]
        qlength = query.shape[1]
        kvlength = key.shape[1]
        num_heads = query.shape[2]
        head_dim = query.shape[3]

        # block reshape before attention
        num_q_blocks = qlength // self.block_size
        num_kv_blocks = kvlength // self.block_size

        block_query = jnp.reshape(query, (bs, num_q_blocks, self.block_size, num_heads, head_dim))
        block_key = jnp.reshape(key, (bs, num_kv_blocks, self.block_size, num_heads, head_dim))
        block_value = jnp.reshape(value, (bs, num_kv_blocks, self.block_size, num_heads, head_dim))

        _, attention_bias = make_block_attention_mask(
            seq_shape=key[:-2],
            bs=bs,
            num_query_blocks=num_q_blocks,
            block_size=self.block_size,
            num_heads=num_heads,
            dtype=self.dtype,
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
                dtype=self.dtype,
                bias=attention_bias,
                precision=self.precision,
                dropout_rng=dropout_rng,
                dropout_rate=self.dropout_rate,
                broadcast_dropout=self.broadcast_dropout,
                deterministic=deterministic)

        return jnp.reshape(x, (bs, qlength, num_heads, head_dim))

