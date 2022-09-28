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

"""Performer-based language models."""

from functools import partial
from typing import Any, Optional

from flax import linen as nn
import jax.numpy as jnp

from lra_benchmarks.models.performer import performer_attention
from lra_benchmarks.models.generic import generic, attention
from lra_benchmarks.models.generic.module_collection import ModuleCollection

_ATTENTION_FNS = {
        'dot_product':
                lambda qkv_dim, unidirectional=False: nn.dot_product_attention,
        'softmax':
                performer_attention.make_fast_softmax_attention,
        'generalized':
                performer_attention.make_fast_generalized_attention,
}


class PerformerBlock(nn.Module):
    """Performer layer (https://arxiv.org/abs/2006.03555)."""

    raw_attention_fn: Any
    qkv_dim: int
    mlp_dim: int
    num_heads: int
    dtype: Any=jnp.float32
    dropout_rate: float=0.1
    attention_dropout_rate: float=0.1
    max_len: int=512
    layer_num: int=0

    @nn.compact
    def __call__(self, inputs, *, inputs_segmentation=None, causal_mask: bool=False,
                 padding_mask=None, deterministic: bool=False):

        attention_fn = self.raw_attention_fn(self.qkv_dim // self.num_heads,
                                             unidirectional=causal_mask)

        encoded = generic.GenericBlock(
            attention_module=attention.MaskedSelfAttention,
            qkv_dim=self.qkv_dim,
            mlp_dim=self.mlp_dim,
            num_heads=self.num_heads,
            dtype=self.dtype,
            dropout_rate=self.dropout_rate,
            attention_dropout_rate=self.attention_dropout_rate,
            max_len=self.max_len,
            attention_module_kwargs={"attention_fn": attention_fn},
        )(
            inputs,
            inputs_segmentation=inputs_segmentation,
            causal_mask=causal_mask,
            padding_mask=padding_mask,
            deterministic=deterministic,
        )
        return encoded


def get_modules(attention_fn_cls: str='generalized',
                attention_fn_kwargs: Optional[dict[str, Any]]=None) -> ModuleCollection:

    if isinstance(attention_fn_cls, str):
        raw_attention_fn = _ATTENTION_FNS[attention_fn_cls]
    else:
        raw_attention_fn = attention_fn_cls

    if attention_fn_kwargs is not None:
        raw_attention_fn = partial(raw_attention_fn, **attention_fn_kwargs)

    return ModuleCollection(None, block=partial(PerformerBlock, raw_attention_fn=raw_attention_fn))

