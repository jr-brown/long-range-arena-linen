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

"""Synthesizer Attention modules."""
from functools import partial
from collections.abc import Iterable  # pylint: disable=g-importing-member
from typing import Any

from absl import logging

from flax import linen as nn
import jax
from jax import lax
from jax import random
import jax.numpy as jnp
import jax.nn as jnn

import numpy as onp


def synthetic_attention(query,
                        key,
                        value,
                        synthetic,
                        dropout_rng,
                        dtype=jnp.float32,
                        bias=None,
                        axis=None,
                        broadcast_dropout=True,
                        dropout_rate=0.,
                        deterministic=False,
                        precision=None,
                        ignore_dot_product=False):
    """Computes dot-product attention given query, key, and value.

    Supports additional synthetic weights mixture.

    This is the core function for applying attention based on
    https://arxiv.org/abs/1706.03762. It calculates the attention weights given
    query and key and combines the values using the attention weights. This
    function supports multi-dimensional inputs.


    Args:
        query: queries for calculating attention with shape of `[batch_size, dim1,
            dim2, ..., dimN, num_heads, mem_channels]`.
        key: keys for calculating attention with shape of `[batch_size, dim1, dim2,
            ..., dimN, num_heads, mem_channels]`.
        value: values to be used in attention with shape of `[batch_size, dim1,
            dim2,..., dimN, num_heads, value_channels]`.
        synthetic: list of weight matrices of [len, len].
        dtype: the dtype of the computation (default: float32)
        bias: bias for the attention weights. This can be used for incorporating
            autoregressive mask, padding mask, proximity bias.
        axis: axises over which the attention is applied.
        broadcast_dropout: bool: use a broadcasted dropout along batch dims.
        dropout_rng: JAX PRNGKey: to be used for dropout
        dropout_rate: dropout rate
        deterministic: bool, deterministic or not (to apply dropout)
        precision: numerical precision of the computation see `jax.lax.Precision`
            for details.
        ignore_dot_product: bool, to ignore dot product or not.

    Returns:
        Output of shape `[bs, dim1, dim2, ..., dimN,, num_heads, value_channels]`.
    """
    if axis is None:
        axis = tuple(range(1, key.ndim - 2))
    if not isinstance(axis, Iterable):
        axis = (axis,)
    if not ignore_dot_product:
        assert key.shape[:-1] == value.shape[:-1]
        assert (query.shape[0:1] == key.shape[0:1] and
                        query.shape[-1] == key.shape[-1])
        assert key.ndim == query.ndim
        assert key.ndim == value.ndim
        for ax in axis:
            if not (query.ndim >= 3 and 1 <= ax < query.ndim - 2):
                raise ValueError('Attention axis must be between the batch '
                                                  'axis and the last-two axes.')
        depth = query.shape[-1]
        n = key.ndim
        # batch_dims is  <bs, <non-attention dims>, num_heads>
        batch_dims = tuple(onp.delete(range(n), axis + (n - 1,)))
        # q & k -> (bs, <non-attention dims>, num_heads, <attention dims>, channels)
        qk_perm = batch_dims + axis + (n - 1,)
        key = key.transpose(qk_perm)
        query = query.transpose(qk_perm)
        # v -> (bs, <non-attention dims>, num_heads, channels, <attention dims>)
        v_perm = batch_dims + (n - 1,) + axis
        value = value.transpose(v_perm)

        query = query / jnp.sqrt(depth).astype(dtype)
        batch_dims_t = tuple(range(len(batch_dims)))

        attn_weights = lax.dot_general(
                query,
                key, (((n - 1,), (n - 1,)), (batch_dims_t, batch_dims_t)),
                precision=precision)
    else:
        n = key.ndim
        batch_dims = tuple(onp.delete(range(n), axis + (n - 1,)))
        v_perm = batch_dims + (n - 1,) + axis
        qk_perm = batch_dims + axis + (n - 1,)
        value = value.transpose(v_perm)
        batch_dims_t = tuple(range(len(batch_dims)))
        attn_weights = 0

    if synthetic:
        # add synthetic attention
        for syn_weights in synthetic:
            attn_weights += syn_weights

    # apply attention bias: masking, droput, proximity bias, ect.
    if bias is not None:
        attn_weights = attn_weights + bias

    # normalize the attention weights
    norm_dims = tuple(range(attn_weights.ndim - len(axis), attn_weights.ndim))
    attn_weights = jax.nn.softmax(attn_weights, axis=norm_dims)
    attn_weights = attn_weights.astype(dtype)

    # apply dropout
    if not deterministic and dropout_rate > 0.:
        keep_prob = jax.lax.tie_in(attn_weights, 1.0 - dropout_rate)
        if broadcast_dropout:
            # dropout is broadcast across the batch+head+non-attention dimension
            dropout_dims = attn_weights.shape[-(2 * len(axis)):]
            dropout_shape = (tuple([1] * len(batch_dims_t)) + dropout_dims)
            keep = random.bernoulli(dropout_rng, keep_prob, dropout_shape)
        else:
            keep = random.bernoulli(dropout_rng, keep_prob, attn_weights.shape)
        multiplier = (keep.astype(attn_weights.dtype) /
                                    jnp.asarray(keep_prob, dtype=dtype))
        attn_weights = attn_weights * multiplier

    # compute the new values given the attention weights
    wv_contracting_dims = (norm_dims, range(value.ndim - len(axis), value.ndim))
    y = lax.dot_general(
            attn_weights,
            value, (wv_contracting_dims, (batch_dims_t, batch_dims_t)),
            precision=precision)

    # back to (bs, dim1, dim2, ..., dimN, num_heads, channels)
    perm_inv = _invert_perm(qk_perm)
    y = y.transpose(perm_inv)
    return y


def _invert_perm(perm):
    perm_inv = [0] * len(perm)
    for i, j in enumerate(perm):
        perm_inv[j] = i
    return tuple(perm_inv)


class SynthesizerAttention(nn.Module):
    """Multi-head Synthesizer Architecture."""

    ignore_dot_product: bool
    synthesizer_mode: str
    k: int
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
            attention_axis: axes over which the attention is applied ( 'None' means
                attention over all axes, but batch, heads, and features).
            causal_mask: boolean specifying whether to apply a causal mask on the
                attention weights. If True, the output at timestep `t` will not depend
                on inputs at timesteps strictly greater than `t`.
            padding_mask: boolean specifying query tokens that are pad token.
            key_padding_mask: boolean specifying key-value tokens that are pad token.
            segmentation: segment indices for packed inputs_q data.
            key_segmentation: segment indices for packed inputs_kv data.
            cache: an instance of `flax.nn.attention.Cache` used for efficient
                autoregressive decoding.
            broadcast_dropout: bool: use a broadcasted dropout along batch dims.
            dropout_rng: JAX PRNGKey: to be used for dropout
            dropout_rate: dropout rate
            deterministic: bool, deterministic or not (to apply dropout)
            precision: numerical precision of the computation see `jax.lax.Precision`
                for details.
            kernel_init: initializer for the kernel of the Dense layers.
            bias_init: initializer for the bias of the Dense layers.
            bias: bool: whether pointwise QKVO dense transforms use bias.
            max_len: int, the maximum supported sequence length.
            ignore_dot_product: bool, to ignore the dot product attention or not.
            synthesizer_mode: str support 'dense' and 'random' or 'dense+random'
            k: int, low rank factorized attention.

        Returns:
            output of shape `[bs, dim1, dim2, ..., dimN, features]`.
        """

        assert inputs_q.ndim == 3

        if inputs_kv is None:
            inputs_kv = inputs_q

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
        qlength = inputs_q.shape[-2]
        kvlength = inputs_kv.shape[-2]

        if self.ignore_dot_product:
            value = dense(name='value')(inputs_kv)
            key = value
            query = inputs_q
        else:
            qd, kd, vd = dense(name='query'), dense(name='key'), dense(name='value')
            query, key, value = qd(inputs_q), kd(inputs_kv), vd(inputs_kv)

        syn_weights_list = []
        # logging.info(self.synthesizer_mode)
        if 'random' in self.synthesizer_mode:
            if 'factorized_random' in self.synthesizer_mode:
                # logging.info('Using factorized random')
                rand_syn_weights1 = self.param('random1', self.kernel_init,
                                               (self.num_heads, self.max_len, self.k))
                rand_syn_weights2 = self.param('random2', self.kernel_init,
                                               (self.num_heads, self.k, self.max_len))
                rand_syn_weights1 = rand_syn_weights1[:, :qlength, :]
                rand_syn_weights2 = rand_syn_weights2[:, :, :kvlength]
                rand_syn_weights = jnp.einsum('hlk,hkn->hln', rand_syn_weights1, rand_syn_weights2)
                rand_syn_weights = jax.lax.broadcast(rand_syn_weights, (inputs_q.shape[0],))
                syn_weights_list.append(rand_syn_weights)
            else:
                rand_syn_weights = self.param('random', self.kernel_init,
                                              (self.num_heads, self.max_len, self.max_len))
                rand_syn_weights = rand_syn_weights[:, :qlength, :kvlength]
                rand_syn_weights = jax.lax.broadcast(rand_syn_weights, (inputs_q.shape[0],))
                syn_weights_list.append(rand_syn_weights)
        if 'dense' in self.synthesizer_mode:
            dense_syn = nn.DenseGeneral(
                features=(self.num_heads, head_dim),
                kernel_init=self.kernel_init,
                bias_init=self.bias_init,
                use_bias=self.bias,
                precision=self.precision,
                name='dense_syn',
                dtype=self.dtype)
            dense_syn_length = nn.Dense(
                features=self.max_len,
                kernel_init=self.kernel_init,
                bias_init=self.bias_init,
                use_bias=self.bias,
                precision=self.precision,
                name='dense_syn_len',
                dtype=self.dtype)
            proj = dense_syn(inputs_q)
            proj = jnn.relu(proj)
            proj = dense_syn_length(proj)
            # TODO(yitay) check if this reshape is needed
            dense_syn_weights = proj.reshape((inputs_q.shape[0], self.num_heads,
                                              qlength, self.max_len))
            dense_syn_weights = dense_syn_weights[:, :, :, :qlength]
            syn_weights_list.append(dense_syn_weights)

        # create attention masks
        mask_components = []

        if causal_mask:
            mask_components.append(nn.make_causal_mask(key[:-2]))

        if not self.ignore_dot_product:
            if padding_mask is not None:
                if key_padding_mask is None:
                    key_padding_mask = padding_mask
                mask_components.append(nn.make_attention_mask(padding_mask, key_padding_mask))

            if segmentation is not None:
                if key_segmentation is None:
                    key_segmentation = segmentation
                mask_components.append(nn.make_attention_mask(segmentation, key_segmentation,
                                                              pairwise_fn=jnp.equal))

        if mask_components:
            attention_mask = nn.combine_masks(*mask_components)

            # attention mask in the form of attention bias
            attention_bias = lax.select(
                attention_mask > 0,
                jnp.full(attention_mask.shape, 0.).astype(self.dtype),
                jnp.full(attention_mask.shape, -1e10).astype(self.dtype))
        else:
            attention_bias = None

        if self.dropout_rng is None and not deterministic:
            dropout_rng = self.make_rng('dropout')
        else:
            dropout_rng = self.dropout_rng

        # apply attention
        x = synthetic_attention(
                query,
                key,
                value,
                syn_weights_list,
                dropout_rng=dropout_rng,
                dtype=self.dtype,
                bias=attention_bias,
                precision=self.precision,
                dropout_rate=self.dropout_rate,
                broadcast_dropout=self.broadcast_dropout,
                deterministic=deterministic,
                ignore_dot_product=self.ignore_dot_product)

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

class SynthesizerSelfAttention(SynthesizerAttention):
    def __call__(self, inputs, **kwargs):
        return super().__call__(inputs, inputs_kv=None, **kwargs)

