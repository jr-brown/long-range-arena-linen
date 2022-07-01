from typing import Any, Optional

from flax import linen as nn
import jax.numpy as jnp
import jax.nn as jnn

from lra_benchmarks.models.layers import common_layers


class GenericBlock(nn.Module):
    """Generic Layer"""

    attention_module: nn.Module
    qkv_dim: Any
    mlp_dim: Any
    num_heads: Any
    dtype: Any=jnp.float32
    dropout_rate: Any=0.1
    attention_dropout_rate: Any=0.1
    attention_module_kwargs: Optional[dict]=None

    @nn.compact
    def __call__(self, inputs, *, inputs_segmentation=None, causal_mask: bool=False,
                 padding_mask=None, deterministic: bool=False,
                 attention_kwargs: Optional[dict[str, Any]]=None):
        """Applies GenericBlock module.

        Args:
            TODO

        Returns:
            Output after the block

        """

        if attention_kwargs is None:
            attention_kwargs = {}

        if self.attention_module_kwargs is None:
            attention_module_kwargs = {}
        else:
            attention_module_kwargs = self.attention_module_kwargs

        # Attention block.
        assert inputs.ndim == 3
        x = nn.LayerNorm()(inputs)
        x = self.attention_module(
                num_heads=self.num_heads,
                dtype=self.dtype,
                qkv_features=self.qkv_dim,
                kernel_init=jnn.initializers.xavier_uniform(),
                bias_init=jnn.initializers.normal(stddev=1e-6),
                bias=False,
                broadcast_dropout=False,
                dropout_rate=self.attention_dropout_rate,
                **attention_module_kwargs
        )(x, segmentation=inputs_segmentation,causal_mask=causal_mask, padding_mask=padding_mask,
          deterministic=deterministic, **attention_kwargs)
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

