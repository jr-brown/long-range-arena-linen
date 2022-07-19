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
"""Performer-based language models."""
from functools import partial
from typing import Any

from flax import linen as nn
import jax.numpy as jnp

from lra_benchmarks.models.performer import performer_attention
from lra_benchmarks.models.generic import generic, attention

_ATTENTION_FNS = {
        'dot_product':
                lambda qkv_dim, unidirectional=False: nn.dot_product_attention,
        'softmax':
                performer_attention.make_fast_softmax_attention,
        'generalized':
                performer_attention.make_fast_generalized_attention,
}
_DEFAULT_ATTENTION_FN_CLS = 'generalized'


class PerformerBlock(nn.Module):
    """Performer layer (https://arxiv.org/abs/2006.03555)."""

    qkv_dim: int
    mlp_dim: int
    num_heads: int
    dtype: Any=jnp.float32
    dropout_rate: float=0.1
    attention_dropout_rate: float=0.1
    max_len: int=512
    attention_fn_cls: Any=_DEFAULT_ATTENTION_FN_CLS
    attention_fn_kwargs: Any=None
    block_size: int=50
    layer_num: int=0

    @nn.compact
    def __call__(self, inputs, *, inputs_segmentation=None, causal_mask: bool=False,
                 padding_mask=None, deterministic: bool=False):

        def _make_attention_fn(attention_fn_cls, attention_fn_kwargs=None):
            attention_fn = (_ATTENTION_FNS[attention_fn_cls]
                            if isinstance(attention_fn_cls, str)
                            else attention_fn_cls)
            return (attention_fn if attention_fn_kwargs is None else partial(
                    attention_fn, **attention_fn_kwargs))

        attention_fn = _make_attention_fn(self.attention_fn_cls, self.attention_fn_kwargs)(
            self.qkv_dim // self.num_heads, unidirectional=causal_mask)

        encoded = generic.GenericBlock(
            attention_module=attention.MaskedSelfAttention,
            qkv_dim=self.qkv_dim,
            mlp_dim=self.mlp_dim,
            num_heads=self.num_heads,
            dtype=self.dtype,
            dropout_rate=self.dropout_rate,
            attention_dropout_rate=self.attention_dropout_rate,
            max_len=self.max_len,
            block_size=self.block_size,
            attention_module_kwargs={"attention_fn": attention_fn},
        )(
            inputs,
            inputs_segmentation=inputs_segmentation,
            causal_mask=causal_mask,
            padding_mask=padding_mask,
            deterministic=deterministic,
        )
        return encoded


class PerformerEncoder(nn.Module):
    """Local Transformer Encoder."""

    vocab_size: Any
    shared_embedding: Any=None
    use_bfloat16: bool=False
    dtype: Any=jnp.float32
    emb_dim: int=512
    num_heads: int=8
    num_layers: int=6
    qkv_dim: int=512
    mlp_dim: int=2048
    max_len: int=512
    dropout_rate: float=0.1
    attention_dropout_rate: float=0.1
    learn_pos_emb: bool=False
    classifier: bool=False
    classifier_pool: str='CLS'
    num_classes: int=10
    attention_fn_cls: Any=_DEFAULT_ATTENTION_FN_CLS
    attention_fn_kwargs: Any=None

    def setup(self):
        if self.classifier and self.classifier_pool == 'CLS':
            self._max_len = self.max_len + 1
        else:
            self._max_len = self.max_len

    @nn.compact
    def __call__(self, inputs, *, inputs_positions=None, inputs_segmentation=None, train=True):
        block_module_kwargs = {
            "attention_fn_cls": self.attention_fn_cls,
            "attention_fn_kwargs": self.attention_fn_kwargs,
        }
        x = generic.GenericEncoder(
            block_module=PerformerBlock,
            vocab_size=self.vocab_size,
            shared_embedding=self.shared_embedding,
            use_bfloat16=self.use_bfloat16,
            dtype=self.dtype,
            emb_dim=self.emb_dim,
            num_heads=self.num_heads,
            num_layers=self.num_layers,
            qkv_dim=self.qkv_dim,
            mlp_dim=self.mlp_dim,
            max_len=self._max_len,
            dropout_rate=self.dropout_rate,
            attention_dropout_rate=self.attention_dropout_rate,
            learn_pos_emb=self.learn_pos_emb,
            classifier=self.classifier,
            classifier_pool=self.classifier_pool,
            num_classes=self.num_classes,
            block_module_kwargs=block_module_kwargs
        )(inputs, inputs_positions=inputs_positions, inputs_segmentation=inputs_segmentation,
          train=train)
        return x


class PerformerDualEncoder(nn.Module):
    """Performer Model for Matching (dual encoding) tasks."""

    vocab_size: Any=None
    use_bfloat16: bool=False
    emb_dim: int=512
    num_heads: int=8
    num_layers: int=6
    qkv_dim: int=512
    mlp_dim: int=2048
    max_len: int=2048
    dropout_rate: float=0.1
    attention_dropout_rate: float=0.1
    classifier: bool=True
    classifier_pool: str='CLS'
    num_classes: int=2
    interaction: Any=None
    attention_fn_cls: Any=_DEFAULT_ATTENTION_FN_CLS
    attention_fn_kwargs: Any=None

    @nn.compact
    def __call__(self, inputs1, inputs2, *, inputs1_positions=None, inputs2_positions=None,
                 inputs1_segmentation=None, inputs2_segmentation=None, train: bool=False):
        encoder_module_kwargs = {
            "attention_fn_cls": self.attention_fn_cls,
            "attention_fn_kwargs": self.attention_fn_kwargs,
        }
        encoded = generic.GenericDualEncoder(
            encoder_module=PerformerEncoder,
            vocab_size=self.vocab_size,
            use_bfloat16=self.use_bfloat16,
            emb_dim=self.emb_dim,
            num_heads=self.num_heads,
            num_layers=self.num_layers,
            qkv_dim=self.qkv_dim,
            mlp_dim=self.mlp_dim,
            max_len=self.max_len,
            dropout_rate=self.dropout_rate,
            attention_dropout_rate=self.attention_dropout_rate,
            classifier=self.classifier,
            classifier_pool=self.classifier_pool,
            num_classes=self.num_classes,
            interaction=self.interaction,
            encoder_module_kwargs=encoder_module_kwargs,
        )(
            inputs1, inputs2,
            inputs1_positions=inputs1_positions,
            inputs2_positions=inputs2_positions,
            inputs1_segmentation=inputs1_segmentation,
            inputs2_segmentation=inputs2_segmentation,
            train=train,
        )
        return encoded


class PerformerDecoder(nn.Module):
    """Performer Decoder."""

    vocab_size: Any
    emb_dim: int=512
    num_heads: int=8
    num_layers: int=6
    qkv_dim: int=512
    mlp_dim: int=2048
    max_len: int=2048
    shift: bool=True
    dropout_rate: float=0.1
    attention_dropout_rate: float=0.1
    attention_fn_cls: Any=_DEFAULT_ATTENTION_FN_CLS
    attention_fn_kwargs: Any=None

    @nn.compact
    def __call__(self, inputs, *, train: bool=False):
        block_module_kwargs = {
            "attention_fn_cls": self.attention_fn_cls,
            "attention_fn_kwargs": self.attention_fn_kwargs,
        }
        logits = generic.GenericDecoder(
            block_module=PerformerBlock,
            vocab_size=self.vocab_size,
            emb_dim=self.emb_dim,
            num_heads=self.num_heads,
            num_layers=self.num_layers,
            qkv_dim=self.qkv_dim,
            mlp_dim=self.mlp_dim,
            max_len=self.max_len,
            shift=self.shift,
            dropout_rate=self.dropout_rate,
            attention_dropout_rate=self.attention_dropout_rate,
            block_module_kwargs=block_module_kwargs,
        )(inputs, train=train)
        return logits

