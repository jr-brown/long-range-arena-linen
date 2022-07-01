from typing import Any

import flax.linen as nn
import jax.numpy as jnp
import jax.nn as jnn

class MaskedSelfAttention(nn.Module):

    num_heads: Any
    dtype: Any=jnp.float32
    qkv_features: Any=None
    kernel_init: Any=nn.Dense.kernel_init
    bias_init: Any=jnn.initializers.zeros
    bias: Any=True
    broadcast_dropout: Any=True
    dropout_rate: Any=0.

    @nn.compact
    def __call__(self, x, *, segmentation=None, causal_mask: bool=False, padding_mask=None, deterministic: bool=False):

        mask = nn.make_attention_mask(padding_mask, padding_mask)

        if causal_mask:
            mask = nn.combine_masks(mask, nn.make_causal_mask(x))

        if segmentation is not None:
            raise Exception("Not implemented yet")

        x = nn.SelfAttention(
                num_heads=self.num_heads,
                dtype=self.dtype,
                qkv_features=self.qkv_features,
                kernel_init=jnn.initializers.xavier_uniform(),
                bias_init=jnn.initializers.normal(stddev=1e-6),
                use_bias=False,
                broadcast_dropout=False,
                dropout_rate=self.dropout_rate,
        )(x, deterministic=deterministic, mask=mask)

        return x

