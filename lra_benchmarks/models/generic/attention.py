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

from typing import Any, Callable
from absl import logging
from pprint import pformat
from functools import partial

import json

import jax.numpy as jnp
import jax.nn as jnn

import flax.linen as nn
from flax.linen.dtypes import promote_dtype

from lra_benchmarks.utils.array_utils import make_attention_mask, convert_array_to_list


def dot_product_attention_with_weight_logging(
    query, key, value, bias=None, mask=None, broadcast_dropout=True, dropout_rng=None,
    dropout_rate=0, deterministic=False, dtype=None, precision=None,
    output_db_path=None,
):

    """
    Copy of flax.linen.attention.dot_product_attention but logs attention weights

    Don't use in a parallel computation since that will just print tracers
    """

    query, key, value = promote_dtype(query, key, value, dtype=dtype)
    dtype = query.dtype
    assert key.ndim == query.ndim == value.ndim, 'q, k, v must have same rank.'
    assert query.shape[:-3] == key.shape[:-3] == value.shape[:-3], (
            'q, k, v batch dims must match.')
    assert query.shape[-2] == key.shape[-2] == value.shape[-2], (
            'q, k, v num_heads must match.')
    assert key.shape[-3] == value.shape[-3], 'k, v lengths must match.'

    # compute attention weights
    attn_weights = nn.attention.dot_product_attention_weights(
            query, key, bias, mask, broadcast_dropout, dropout_rng, dropout_rate,
            deterministic, dtype, precision)

    logging.info("Attention weights being saved, shape:\n" + pformat(attn_weights.shape))

    with open(output_db_path) as f:
        output_db = json.load(f)

    # Dims will be [batch, num_heads, q_length, kv_length]
    # Therefore attn_weights[0][0][x][y] is how important token x is to understanding token y in batch 0 and head 0
    output_db["attn_weights"] = convert_array_to_list(attn_weights)

    with open(output_db_path, 'w', encoding="utf-8") as f:
        json.dump(output_db, f, ensure_ascii=False, indent=4)


    # return weighted sum over values for each query position
    return jnp.einsum('...hqk,...khd->...qhd', attn_weights, value,
                                        precision=precision)


class MaskedSelfAttention(nn.Module):

    num_heads: Any
    dtype: Any=jnp.float32
    qkv_features: Any=None
    kernel_init: Any=nn.Dense.kernel_init
    bias_init: Any=jnn.initializers.zeros
    bias: Any=True
    broadcast_dropout: Any=True
    dropout_rate: Any=0.
    max_len: int=512
    block_size: int=50
    layer_num: int=0
    attention_fn: Callable[[Any, Any, Any], Any]=nn.dot_product_attention

    @nn.compact
    def __call__(self, x, *, segmentation=None, causal_mask: bool=False, padding_mask=None,
                 deterministic: bool=False, log_attention_weights=False, log_output_db_path=None):

        _, mask = make_attention_mask(
            seq_shape=x.shape[:-2],
            dtype=self.dtype,
            causal_mask=causal_mask,
            padding_mask=padding_mask,
            segmentation=segmentation,
            use_attention_bias=False
        )

        if log_attention_weights and self.attention_fn == nn.dot_product_attention:
            attention_fn = partial(dot_product_attention_with_weight_logging,
                                   output_db_path=log_output_db_path)
        else:
            attention_fn = self.attention_fn

        x = nn.SelfAttention(
                num_heads=self.num_heads,
                dtype=self.dtype,
                qkv_features=self.qkv_features,
                kernel_init=jnn.initializers.xavier_uniform(),
                bias_init=jnn.initializers.normal(stddev=1e-6),
                use_bias=False,
                broadcast_dropout=False,
                dropout_rate=self.dropout_rate,
                attention_fn=attention_fn,
        )(x, deterministic=deterministic, mask=mask)

        return x

