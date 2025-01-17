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
#
# NOTICE
# Modified by Jason Brown 2022

"""Local Attention modules."""

from functools import partial
from typing import Any
from math import ceil

from flax import linen as nn

import jax.numpy as jnp
import jax.nn as jnn

from lra_benchmarks.utils.array_utils import make_block_attention_mask, pad_inputs


class LocalAttention(nn.Module):
    """Multi-head Local Attention Architecture."""

    block_size: int
    num_heads: Any
    dtype: Any=jnp.float32
    qkv_features: Any=None
    out_features: Any=None
    broadcast_dropout: Any=True
    dropout_rng: Any=None
    dropout_rate: Any=0.
    precision: Any=None
    kernel_init: Any=nn.Dense.kernel_init
    bias_init: Any=jnn.initializers.zeros
    bias: Any=True
    max_len: int=512
    layer_num: int=0

    def setup(self):
        self.n_blocks = ceil(self.max_len / self.block_size)
        self.blocks_total_len = self.n_blocks * self.block_size

    @nn.compact
    def __call__(self, inputs_q, inputs_kv=None, *, segmentation=None, key_segmentation=None,
                 causal_mask: bool=False, padding_mask=None, key_padding_mask=None,
                 deterministic: bool=False):
        """Applies multi-head synthesizer attention on the input data.

        Projects the inputs into multi-headed query, key, and value vectors,
        applies dot-product attention and project the results to an output vector.

        This can be used for encoder-decoder attention by specifying both `inputs_q`
        and `inputs_kv` orfor self-attention by only specifying `inputs_q` and
        setting `inputs_kv` to None.

        Args:
            inputs_q: input queries of shape `[bs, dim1, dim2, ..., dimN, features]`.
            inputs_kv: key/values of shape `[bs, dim1, dim2, ..., dimN, features]`
                or None for self-attention, inn which case key/values will be derived
                from inputs_q.
            num_heads: number of attention heads. Features (i.e. inputs_q.shape[-1])
                should be divisible by the number of heads.
            dtype: the dtype of the computation (default: float32)
            qkv_features: dimension of the key, query, and value.
            out_features: dimension of the last projection
            causal_mask: boolean specifying whether to apply a causal mask on the
                attention weights. If True, the output at timestep `t` will not depend
                on inputs at timesteps strictly greater than `t`.
            padding_mask: boolean specifying query tokens that are pad token.
            key_padding_mask: boolean specifying key-value tokens that are pad token.
            broadcast_dropout: bool: use a broadcasted dropout along batch dims.
            dropout_rng: JAX PRNGKey: to be used for dropout
            dropout_rate: dropout rate
            deterministic: bool, deterministic or not (to apply dropout)
            precision: numerical precision of the computation see `jax.lax.Precision`
                for details.
            kernel_init: initializer for the kernel of the Dense layers.
            bias_init: initializer for the bias of the Dense layers.
            bias: bool: whether pointwise QKVO dense transforms use bias.
            block_size: int, block size.

        Returns:
            output of shape `[bs, dim1, dim2, ..., dimN, features]`.
        """

        assert inputs_q.ndim == 3
        orig_len = inputs_q.shape[-2]

        inputs_q, inputs_kv, padding_mask = pad_inputs(orig_len, self.blocks_total_len, inputs_q,
                                                       inputs_kv, padding_mask)

        features = self.out_features or inputs_q.shape[-1]
        qkv_features = self.qkv_features or inputs_q.shape[-1]

        assert qkv_features % self.num_heads == 0, (
            'Memory dimension must be divisible by number of heads.')
        head_dim = qkv_features // self.num_heads

        dense = partial(nn.DenseGeneral,
                        axis=-1,
                        features=(self.num_heads, head_dim),
                        kernel_init=self.kernel_init,
                        bias_init=self.bias_init,
                        use_bias=self.bias,
                        dtype=self.dtype,
                        precision=self.precision)

        # project inputs_q to multi-headed q/k/v
        # dimensions are then [bs, dims..., n_heads, n_features_per_head]
        qd, kd, vd = dense(name='query'), dense(name='key'), dense(name='value')
        query, key, value = qd(inputs_q), kd(inputs_kv), vd(inputs_kv)

        qlength = inputs_q.shape[-2]
        bs = inputs_q.shape[0]
        kvlength = inputs_kv.shape[-2]

        # block reshape before attention
        num_query_blocks = qlength // self.block_size
        num_kv_blocks = kvlength // self.block_size

        block_query = jnp.reshape(
                query, (bs, num_query_blocks, self.block_size, self.num_heads, head_dim))
        block_key = jnp.reshape(
                key, (bs, num_kv_blocks, self.block_size, self.num_heads, head_dim))
        block_value = jnp.reshape(
                value, (bs, num_kv_blocks, self.block_size, self.num_heads, head_dim))

        _, attention_bias = make_block_attention_mask(
            seq_shape=key[:-2],
            bs=bs,
            num_query_blocks=num_query_blocks,
            block_size=self.block_size,
            num_heads=self.num_heads,
            dtype=self.dtype,
            causal_mask=causal_mask,
            padding_mask=padding_mask,
            key_padding_mask=key_padding_mask,
            segmentation=segmentation,
            key_segmentation=key_segmentation,
            use_attention_bias=True,
        )

        if self.dropout_rng is None and not deterministic:
            dropout_rng = self.make_rng('dropout')
        else:
            dropout_rng = self.dropout_rng

        # apply attention
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

        x = jnp.reshape(x, (bs, qlength, self.num_heads, head_dim))

        # back to the original inputs dimensions
        out = nn.DenseGeneral(
                features=features,
                axis=(-2, -1),
                kernel_init=self.kernel_init,
                bias_init=self.bias_init,
                use_bias=self.bias,
                dtype=self.dtype,
                precision=self.precision,
                name='out')(x)

        out = out[:, :orig_len, :]

        return out

class LocalSelfAttention(LocalAttention):
    def __call__(self, inputs, **kwargs):
        return super().__call__(inputs, inputs_kv=None, **kwargs)

