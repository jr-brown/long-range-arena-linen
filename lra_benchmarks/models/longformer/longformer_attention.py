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

"""Implements Longformer's attention (https://arxiv.org/abs/2004.05150).

Like the current (8/28/20) Huggingface version, we do not support
dilated and autoregressive attention patterns as they require custom CUDA
kernels to be efficient. "Sliding window" and "global" attention patterns are
supported, however.
"""

from functools import partial
from typing import Any

import numpy as np
from flax import linen as nn

import jax.numpy as jnp
import jax.nn as jnn

from lra_benchmarks.utils.array_utils import make_attention_mask



def _build_global_mask(mask):
    """Builds mask for global attention pattern.

    Args:
        mask: boolean jax array of shape `[batch_size, seq_len]`.

    Returns:
        mask, boolean jax array of shape `[batch_size, 1 (n_heads), seq_len,
        seq_len]`.
    """
    return jnp.logical_or(mask[:, jnp.newaxis, :, jnp.newaxis],
                          mask[:, jnp.newaxis, jnp.newaxis, :])


def _build_sliding_window_mask(window_size, global_mask):
    """Builds mask for sliding window pattern.

    Args:
        window_size: int, size of sliding window.
        global_mask: boolean jax array of shape `[batch_size, seq_len]`.

    Returns:
        mask, boolean jax array of shape `[batch_size, 1 (n_heads), seq_len,
        seq_len]`.

    If `window_size` is odd, both left and right sides have the same receptive
    field. Otherwise, the left side gets one more. Note - we need global mask
    because
    due to the symmetry requirement, non-global positions can still attend to
    global positions.
    """
    seq_len = global_mask.shape[1]
    right_size = window_size // 2
    left_size = window_size - right_size
    left_mask = sum(np.eye(seq_len, k=-i) for i in range(left_size))
    right_mask = sum(np.eye(seq_len, k=i) for i in range(1, right_size + 1))
    mask = left_mask + right_mask
    mask = jnp.array(mask[np.newaxis, np.newaxis, :, :]).astype(jnp.bool_)
    return jnp.logical_or(mask, _build_global_mask(global_mask))


def _get_attention_result(query,
                          key,
                          value,
                          dtype,
                          precision,
                          dropout_rng,
                          dropout_rate,
                          broadcast_dropout,
                          deterministic,
                          mask=None,
                          padding_mask=None,
                          key_padding_mask=None,
                          segmentation=None,
                          key_segmentation=None,
                          apply_causal_mask=False):
    """Helper function returning `[batch_size, seq_len, heads, features]` output."""
    # assumes query/key/value has shape `[batch_size, seq_len, heads, features]`.

    _, attention_bias = make_attention_mask(
        seq_shape=query.shape[:-2],
        dtype=dtype,
        causal_mask=apply_causal_mask,
        padding_mask=padding_mask,
        key_padding_mask=key_padding_mask,
        segmentation=segmentation,
        key_segmentation=key_segmentation,
        use_attention_bias=True,
        base_mask=mask
    )

    return nn.dot_product_attention(
            query,
            key,
            value,
            dtype=dtype,
            bias=attention_bias,
            precision=precision,
            dropout_rng=dropout_rng,
            dropout_rate=dropout_rate,
            broadcast_dropout=broadcast_dropout,
            deterministic=deterministic)


class LongformerAttention(nn.Module):
    """Module implementing Longformer attention."""

    num_heads: Any
    sliding_window_size: Any=512
    dtype: Any=jnp.float32
    qkv_features: Any=None
    out_features: Any=None
    broadcast_dropout: Any=True
    dropout_rng: Any=None
    dropout_rate: Any=0.
    precision: Any=None
    kernel_init: Any=nn.linear.default_kernel_init
    bias_init: Any=jnn.initializers.zeros
    bias: Any=True
    max_len: int=512
    block_size: int=50
    layer_num: int=0

    @nn.compact
    def __call__(self, inputs_q, inputs_kv=None, *, segmentation=None, key_segmentation=None,
                 global_mask=None, causal_mask: bool=False, padding_mask=None,
                 key_padding_mask=None, deterministic: bool=False):
        """Applies longformer multi-head dot product attention on the input data.

        Args:
            inputs_q: input queries of shape `[bs, seq_len, features]`.
            inputs_kv: key/values of shape `[bs, seq_len, features]` or `None` for
                self-attention, in which case key/values will be derived from inputs_q.
            num_heads: number of attention heads (should divide number of features).
            sliding_window_size: size of sliding window attention to use.
            global_mask: boolean matrix of shape `[bs, seq_len]`, where `True`
                indicates that the position is globally attended. By default, no global
                attention is used.
            causal_mask: If true, apply causal attention masking.
            dtype: the dtype of the computation (default: float32).
            qkv_features: dimension of the key, query, and value.
            out_features: dimension of the last projection.
            padding_mask: boolean specifying query tokens that are pad token.
            key_padding_mask: boolean specifying key-value tokens that are pad token.
            segmentation: segment indices for packed inputs_q data.
            key_segmentation: segment indices for packed inputs_kv data.
            broadcast_dropout: use a broadcasted dropout along batch dims.
            dropout_rng: JAX PRNGKey to be use for dropout.
            dropout_rate: dropout rate.
            deterministic: if true, apply dropout, else don't.
            precision: numerical precision of the computation.
            kernel_init: initializer for the kernel of the Dense layers.
            bias_init: initializer for the bias of the Dense layers.
            bias: whether pointwise QKVO dense transforms use bias. query, key, value,
                and returns output of shape `[bs, seq_len, num_heads, value_channels]`.

        Returns:
            output of shape `[bs, seq_len, features]`.
        """
        if inputs_kv is None:
            inputs_kv = inputs_q

        batch_size = inputs_q.shape[0]
        features = self.out_features or inputs_q.shape[-1]
        qkv_features = self.qkv_features or inputs_q.shape[-1]
        seq_len = inputs_q.shape[1]

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

        query_sw = dense(name='query_sliding_window')(inputs_q)
        key_sw = dense(name='key_sliding_window')(inputs_kv)
        value_sw = dense(name='value_sliding_window')(inputs_kv)

        query_global = dense(name='query_global')(inputs_q)
        key_global = dense(name='key_global')(inputs_kv)
        value_global = dense(name='value_global')(inputs_kv)

        if global_mask is None:
            global_mask = jnp.full((batch_size, seq_len), False)

        full_global_mask = _build_global_mask(global_mask)

        sliding_window_mask = _build_sliding_window_mask(
                window_size=self.sliding_window_size, global_mask=global_mask)

        if self.dropout_rng is None and not deterministic:
            dropout_rng = self.make_rng('dropout')
        else:
            dropout_rng = self.dropout_rng

        x_sw = _get_attention_result(
                query=query_sw,
                key=key_sw,
                value=value_sw,
                dtype=self.dtype,
                precision=self.precision,
                dropout_rng=dropout_rng,
                dropout_rate=self.dropout_rate,
                broadcast_dropout=self.broadcast_dropout,
                deterministic=deterministic,
                mask=sliding_window_mask,
                padding_mask=padding_mask,
                key_padding_mask=key_padding_mask,
                segmentation=segmentation,
                key_segmentation=key_segmentation,
                apply_causal_mask=causal_mask)

        x_global = _get_attention_result(
                query=query_global,
                key=key_global,
                value=value_global,
                dtype=self.dtype,
                precision=self.precision,
                dropout_rng=dropout_rng,
                dropout_rate=self.dropout_rate,
                broadcast_dropout=self.broadcast_dropout,
                deterministic=deterministic,
                mask=full_global_mask,
                padding_mask=padding_mask,
                key_padding_mask=key_padding_mask,
                segmentation=segmentation,
                key_segmentation=key_segmentation,
                apply_causal_mask=causal_mask)

        x = jnp.where(global_mask[:, :, jnp.newaxis, jnp.newaxis], x_global, x_sw)

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

        return out

class LongformerSelfAttention(LongformerAttention):
    def __call__(self, inputs, **kwargs):
        return super().__call__(inputs, inputs_kv=None, **kwargs)

