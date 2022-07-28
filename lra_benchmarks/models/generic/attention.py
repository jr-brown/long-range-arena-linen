from typing import Any, Callable, Tuple, Optional
from functools import partial

import flax.linen as nn
from flax.linen.linear import PrecisionLike, default_kernel_init

import jax.numpy as jnp
import jax.nn as jnn

from lra_benchmarks.utils.array_utils import make_attention_mask, pad_inputs


PRNGKey = Any
Shape = Tuple[int, ...]
Dtype = Any
Array = Any


def masked_dot_product_attention(query, key, value,
                                 causal_mask: bool=False,
                                 padding_mask=None,
                                 key_padding_mask=None,
                                 segmentation=None,
                                 key_segmentation=None,
                                 use_attention_bias: bool=False,
                                 dropout_rng=None,
                                 dropout_rate: float=0,
                                 broadcast_dropout: bool=True,
                                 deterministic: bool=False,
                                 dtype=None,
                                 precision=None):

    # Input dimensions are [batch_size, length, num_heads, n_features_per_head]
    assert query.ndim == 4
    assert query.shape == key.shape == value.shape

    _, mask = make_attention_mask(
        seq_shape=query.shape[:-2],
        dtype=dtype,
        causal_mask=causal_mask,
        padding_mask=padding_mask,
        key_padding_mask=key_padding_mask,
        segmentation=segmentation,
        key_segmentation=key_segmentation,
        use_attention_bias=use_attention_bias
    )

    out = nn.dot_product_attention(
        query, key, value,
        mask=mask,
        dropout_rng=dropout_rng,
        dropout_rate=dropout_rate,
        broadcast_dropout=broadcast_dropout,
        deterministic=deterministic,
        dtype=dtype,
        precision=precision,
    )

    return out


class GenericAttention(nn.Module):
    """Generic Attention"""

    num_heads: int
    dtype: Optional[Dtype]=None
    param_dtype: Dtype=jnp.float32
    qkv_features: Optional[int]=None
    out_features: Optional[int]=None
    broadcast_dropout: bool=True
    dropout_rate: float=0
    precision: PrecisionLike=None
    kernel_init: Callable[[PRNGKey, Shape, Dtype], Array]=default_kernel_init
    bias_init: Callable[[PRNGKey, Shape, Dtype], Array]=jnn.initializers.zeros
    use_bias: bool=True
    max_len: int=512
    layer_num: int=0
    attention_fn: Callable[..., Array]=masked_dot_product_attention
    padded_length_fn: Optional[Callable[[nn.Module], int]]=None
    attention_fn_kwargs: Optional[dict[str, Any]]=None

    def setup(self):
        if self.padded_length_fn is not None:
            self.padded_length = self.padded_length_fn(self)
        else:
            self.padded_length = None

    @nn.compact
    def __call__(self, inputs_q, inputs_kv, *, segmentation=None, key_segmentation=None,
                 causal_mask: bool=False, padding_mask=None, key_padding_mask=None,
                 deterministic: bool=False,
                 attention_fn_extra_kwargs: Optional[dict[str, Any]]=None):
        """Applies multi-head dot product attention on the input data.

        Projects the inputs into multi-headed query, key, and value vectors,
        applies dot-product attention and project the results to an output vector.

        Args:
            inputs_q: input queries of shape
                `[batch_sizes, length, features]`.
            inputs_kv: key/values of shape
                `[batch_sizes, length, features]`.
            mask: attention mask of shape
                `[batch_sizes, num_heads, query_length, key/value_length]`.
                Attention weights are masked out if their corresponding mask value
                is `False`.
            deterministic: if false, the attention weight is masked randomly
                using dropout, whereas if true, the attention weights
                are deterministic.

        Returns:
            output of shape `[batch_sizes..., length, features]`.
        """

        assert inputs_q.ndim == 3
        assert inputs_q.shape == inputs_kv.shape
        original_length = inputs_q.shape[1]

        if self.padded_length is not None:
            inputs_q, inputs_kv, padding_mask = pad_inputs(self.padded_length,
                                                           inputs_q, inputs_kv, padding_mask)

        features = self.out_features or inputs_q.shape[-1]
        qkv_features = self.qkv_features or inputs_q.shape[-1]

        assert qkv_features % self.num_heads == 0, (
            'Memory dimension must be divisible by number of heads.')
        head_dim = qkv_features // self.num_heads

        dense = partial(
            nn.DenseGeneral,
            axis=-1,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            features=(self.num_heads, head_dim),
            kernel_init=self.kernel_init,
            bias_init=self.bias_init,
            use_bias=self.use_bias,
            precision=self.precision
        )

        # project inputs_q to multi-headed q/k/v
        # dimensions are then [batch..., length, n_heads, n_features_per_head]
        qd, kd, vd = dense(name='query'), dense(name='key'), dense(name='value')
        query, key, value = qd(inputs_q), kd(inputs_kv), vd(inputs_kv)

        if self.dropout_rate > 0 and not deterministic:
            dropout_rng = self.make_rng('dropout')
        else:
            dropout_rng = None

        if self.attention_fn_kwargs is None:
            attention_fn_kwargs = {}
        else:
            attention_fn_kwargs = self.attention_fn_kwargs

        if attention_fn_extra_kwargs is None:
            attention_fn_extra_kwargs = {}

        # apply attention
        x = self.attention_fn(
            query, key, value,
            causal_mask=causal_mask,
            padding_mask=padding_mask,
            key_padding_mask=key_padding_mask,
            segmentation=segmentation,
            key_segmentation=key_segmentation,
            dropout_rng=dropout_rng,
            dropout_rate=self.dropout_rate,
            broadcast_dropout=self.broadcast_dropout,
            deterministic=deterministic,
            dtype=self.dtype,
            precision=self.precision,
            **attention_fn_kwargs,
            **attention_fn_extra_kwargs,
        )

        # back to the original inputs dimensions
        out = nn.DenseGeneral(
            features=features,
            axis=(-2, -1),
            kernel_init=self.kernel_init,
            bias_init=self.bias_init,
            use_bias=self.use_bias,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            precision=self.precision,
            name='out'
        )(x)

        if self.padded_length is not None:
            out = out[:,:original_length]

        return out


class GenericSelfAttention(GenericAttention):
    def __call__(self, inputs_q, **kwargs):
        return super().__call__(inputs_q, inputs_q, **kwargs)

