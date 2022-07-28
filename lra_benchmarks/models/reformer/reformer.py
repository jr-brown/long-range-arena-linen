"""Reformer language models."""
from typing import Any, Tuple, Callable, Optional
from functools import partial

import flax.linen as nn
from flax.linen.linear import PrecisionLike, default_kernel_init

import jax.numpy as jnp
import jax.nn as jnn

from lra_benchmarks.models.layers import common_layers
from lra_benchmarks.models.reformer import reformer_attention
from lra_benchmarks.models.generic.module_collection import ModuleCollection


PRNGKey = Any
Shape = Tuple[int, ...]
Dtype = Any
Array = Any


def get_modules(chunk_len: int=10, n_chunks_before: int=1, n_hashes: int=1,
                n_buckets: int=10) -> ModuleCollection:

    ReformerAttention = partial(reformer_attention.ReformerAttention,
                                chunk_len=chunk_len, n_chunks_before=n_chunks_before,
                                n_hashes=n_hashes, n_buckets=n_buckets)


    class ReformerBlock(nn.Module):
        qkv_dim: int
        mlp_dim: int
        num_heads: int
        dtype: Any=jnp.float32
        param_dtype: Any=jnp.float32
        dropout_rate: float=0.1
        attention_dropout_rate: float=0.1
        precision: PrecisionLike=None
        kernel_init: Callable[[PRNGKey, Shape, Dtype], Array]=default_kernel_init
        bias_init: Callable[[PRNGKey, Shape, Dtype], Array]=jnn.initializers.zeros
        max_len: int=512
        layer_num: int=0

        @nn.compact
        def __call__(self, inputs, *, inputs_segmentation=None, causal_mask: bool=False,
                     padding_mask=None, deterministic: bool=False):

            # Attention block.
            assert inputs.ndim == 3
            x = nn.LayerNorm()(inputs)

            x = ReformerAttention(
                num_heads=self.num_heads,
                qkv_features=self.qkv_dim,
                dtype=self.dtype,
                broadcast_dropout=False,
                dropout_rate=self.attention_dropout_rate,
                precision=self.precision,
                kernel_init=self.kernel_init,
                bias_init=self.bias_init,
                max_len=self.max_len,
            )(
                inputs, inputs,
                segmentation=inputs_segmentation,
                causal_mask=causal_mask,
                padding_mask=padding_mask, deterministic=deterministic
            )

            x = nn.Dropout(rate=self.dropout_rate)(x, deterministic=deterministic)
            x = x + inputs

            # MLP block.
            y = nn.LayerNorm()(x)
            y = common_layers.MlpBlock(
                mlp_dim=self.mlp_dim,
                dtype=self.dtype,
                dropout_rate=self.dropout_rate,
            )(y, deterministic=deterministic)

            return x + y

    return ModuleCollection(ReformerAttention, block=ReformerBlock)

