from typing import Any, Callable, Tuple, Optional
from functools import partial
from math import ceil

import flax.linen as nn
from flax.linen.linear import PrecisionLike, default_kernel_init

import jax.numpy as jnp
import jax.nn as jnn

from lra_benchmarks.utils.array_utils import make_attention_mask


PRNGKey = Any
Shape = Tuple[int, ...]
Dtype = Any
Array = Any


# Used by many attention mechanisms
def pad_length_fn(x: nn.Module, /, *, block_size: int) -> int:
    return block_size * ceil(x.max_len / block_size)


class MaskedDotProductAttention(nn.Module):

    num_heads: int
    head_dim: int
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

        # Input dimensions are [batch_size, length, num_heads, n_features_per_head]
        assert query.ndim == 4
        assert query.shape == key.shape == value.shape

        _, mask = make_attention_mask(
            seq_shape=query.shape[:-2],
            dtype=self.dtype,
            causal_mask=causal_mask,
            padding_mask=padding_mask,
            key_padding_mask=key_padding_mask,
            segmentation=segmentation,
            key_segmentation=key_segmentation,
            use_attention_bias=self.use_attention_bias
        )

        out = nn.dot_product_attention(
            query, key, value,
            mask=mask,
            dropout_rng=dropout_rng,
            dropout_rate=self.dropout_rate,
            broadcast_dropout=self.broadcast_dropout,
            deterministic=deterministic,
            dtype=self.dtype,
            precision=self.precision,
        )

        return out

