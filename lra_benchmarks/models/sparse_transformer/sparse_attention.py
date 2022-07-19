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
"""Implements Sparse Transformer's attention pattern.

(https://arxiv.org/pdf/1904.10509.pdf).

Note that all attention patterns are causal.
"""
from functools import partial, reduce
from typing import Iterable, Any
import attr

from flax import linen as nn
import jax.numpy as jnp
import jax.nn as jnn

import numpy as np

from lra_benchmarks.utils.array_utils import make_attention_mask


@attr.s
class _Pattern(object):
    pass


@attr.s
class AllPattern(_Pattern):
    pass


@attr.s
class LocalPattern(_Pattern):
    bandwidth = attr.ib()


@attr.s
class StridedPattern(_Pattern):
    stride = attr.ib()


@attr.s
class Fixed1Pattern(_Pattern):
    """Corresponds to the first of two heads in the fixed scheme."""
    block_size = attr.ib()


@attr.s
class Fixed2Pattern(_Pattern):
    """Corresponds to the second of two heads in the fixed scheme."""
    block_size = attr.ib()
    c = attr.ib()


def build_mask(seq_len: int, patterns: Iterable[_Pattern]):
    """Merges multiple attention mask patterns into one."""
    merged_mask = reduce(
            np.logical_or,
            (_build_mask(seq_len, pattern) for pattern in patterns))
    return jnp.array(merged_mask).astype(jnp.bool_)


def _build_mask(n: int, pattern: _Pattern) -> np.ndarray:
    """Helper to build sparse masks."""
    if isinstance(pattern, AllPattern):
        mask = np.tri(n, k=0)
    elif isinstance(pattern, LocalPattern):
        ctx = min(n - 1, pattern.bandwidth - 1)
        mask = sum(np.eye(n, k=-i) for i in range(ctx + 1))
    else:
        r = np.arange(n)
        q = r[:, np.newaxis]
        k = r[np.newaxis, :]
        lower_triangular = k <= q
        if isinstance(pattern, StridedPattern):
            mask = np.remainder(q - k, pattern.stride) == 0
        elif isinstance(pattern, Fixed1Pattern):
            mask = np.floor_divide(q, pattern.block_size) == np.floor_divide(
                    k, pattern.block_size)
        elif isinstance(pattern, Fixed2Pattern):
            remainder = np.remainder(k, pattern.block_size)
            mask = np.logical_or(remainder == 0,
                                                      remainder >= pattern.block_size - pattern.c)
        else:
            raise ValueError('Attention Pattern {} not supported.'.format(pattern))
        mask = np.logical_and(lower_triangular, mask)
    return np.reshape(mask, [1, 1, n, n])


class SparseAttention(nn.Module):
    """Module implementing Sparse Transformer's attention."""

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
    attention_patterns: Any=None
    use_cls_token: bool=False
    block_size: int=50
    layer_num: int=0

    @nn.compact
    def __call__(self, inputs_q, inputs_kv=None, *, segmentation=None, key_segmentation=None,
                 causal_mask: bool=True, padding_mask=None, key_padding_mask=None,
                 deterministic: bool=False):
        """Applies sparse multi-head dot product attention on the input data.

        Args:
            inputs_q: input queries of shape `[bs, seq_len, features]`.
            inputs_kv: key/values of shape `[bs, seq_len, features]` or `None` for
                self-attention, in which case key/values will be derived from inputs_q.
            num_heads: number of attention heads (should divide number of features).
            attention_patterns: list of `_Pattern` objects representing the sparse
                `None`, we use the merged, fixed attention pattern used in the paper for
                EnWik8.
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
            use_cls_token: boolean

        Returns:
            output of shape `[bs, seq_len, features]`.
        """
        del causal_mask  # Always causal

        if inputs_kv is None:
            inputs_kv = inputs_q

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

        # project inputs_q to multi-headed q/k/v
        # dimensions are then [bs, dims..., n_heads, n_features_per_head]
        qd, kd, vd = dense(name='query'), dense(name='key'), dense(name='value')
        query, key, value = qd(inputs_q), kd(inputs_kv), vd(inputs_kv)

        if self.attention_patterns is None:
            # This is the merged fixed attention pattern used in the paper for EnWik8.
            attention_patterns = [
                    Fixed1Pattern(block_size=128),
                    Fixed2Pattern(block_size=128, c=32)
            ]

        if self.use_cls_token:
            # don't mask cls token
            # reset all attention bias to 0 for first position
            mask_seq_len = seq_len - 1
            sparse_mask = build_mask(mask_seq_len, self.attention_patterns)
            shape = sparse_mask.shape
            new_sparse_mask = jnp.full((*shape[:2], shape[2]+1, shape[3]+1), 1.0,
                                       dtype=sparse_mask.dtype)
            sparse_mask = new_sparse_mask.at[:,:,1:,1:].set(sparse_mask)

        else:
            mask_seq_len = seq_len
            sparse_mask = build_mask(mask_seq_len, self.attention_patterns)

        _, attention_bias = make_attention_mask(
            dtype=self.dtype,
            causal_mask=False,
            padding_mask=padding_mask,
            key_padding_mask=key_padding_mask,
            segmentation=segmentation,
            key_segmentation=key_segmentation,
            use_attention_bias=True,
            base_mask=sparse_mask,
        )

        if self.dropout_rng is None and not deterministic:
            dropout_rng = self.make_rng('dropout')
        else:
            dropout_rng = self.dropout_rng

        # apply attention
        x = nn.attention.dot_product_attention(
                query,
                key,
                value,
                dtype=self.dtype,
                bias=attention_bias,
                precision=self.precision,
                dropout_rng=dropout_rng,
                dropout_rate=self.dropout_rate,
                broadcast_dropout=self.broadcast_dropout,
                deterministic=deterministic)

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


class SparseSelfAttention(SparseAttention):
    def __call__(self, inputs, **kwargs):
        return super().__call__(inputs, inputs_kv=None, **kwargs)

