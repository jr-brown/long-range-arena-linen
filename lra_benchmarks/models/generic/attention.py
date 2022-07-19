from typing import Any, Callable

import flax.linen as nn
import jax.numpy as jnp
import jax.nn as jnn

from lra_benchmarks.utils.array_utils import make_attention_mask


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
    attention_fn: Callable[[Any, Any, Any], Any] = nn.dot_product_attention

    @nn.compact
    def __call__(self, x, *, segmentation=None, causal_mask: bool=False, padding_mask=None,
                 deterministic: bool=False):

        _, mask = make_attention_mask(
            seq_shape=x.shape[:-2],
            dtype=self.dtype,
            causal_mask=causal_mask,
            padding_mask=padding_mask,
            segmentation=segmentation,
            use_attention_bias=False
        )

        x = nn.SelfAttention(
                num_heads=self.num_heads,
                dtype=self.dtype,
                qkv_features=self.qkv_features,
                kernel_init=jnn.initializers.xavier_uniform(),
                bias_init=jnn.initializers.normal(stddev=1e-6),
                use_bias=False,
                broadcast_dropout=False,
                dropout_rate=self.dropout_rate,
                attention_fn=self.attention_fn,
        )(x, deterministic=deterministic, mask=mask)

        return x

