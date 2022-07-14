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
"""Sparse Transformer modules."""
from functools import partial
from absl import logging
from typing import Any

from flax import linen as nn
import jax.numpy as jnp

from lra_benchmarks.models.sparse_transformer import sparse_attention
from lra_benchmarks.models.generic import generic


SparseTransformerBlock = partial(generic.GenericBlock,
                                 attention_module=sparse_attention.SparseSelfAttention)


class SparseTransformerEncoder(nn.Module):
    """Local Transformer Encoder."""

    attention_patterns: Any
    vocab_size: Any
    shared_embedding: Any=None
    use_bfloat16: Any=False
    dtype: Any=jnp.float32
    emb_dim: Any=512
    num_heads: Any=8
    num_layers: Any=6
    qkv_dim: Any=512
    mlp_dim: Any=2048
    max_len: Any=512
    dropout_rate: Any=0.1
    attention_dropout_rate: Any=0.1
    learn_pos_emb: Any=False
    classifier: Any=False
    classifier_pool: Any='CLS'
    num_classes: Any=10

    def setup(self):
        if self.classifier and self.classifier_pool == 'CLS':
            self._max_len = self.max_len + 1
        else:
            self._max_len = self.max_len

    @nn.compact
    def __call__(self, inputs, *, inputs_positions=None, inputs_segmentation=None, train=True):
        use_cls_token = False
        if self.classifier_pool == 'CLS':
            use_cls_token = True
            # logging.info('Setting use cls token to true')

        block_module_kwargs = {
            "attention_module_kwargs": {
                "attention_patterns": self.attention_patterns,
                "use_cls_token": use_cls_token,
            }
        }

        x = generic.GenericEncoder(
            block_module=SparseTransformerBlock,
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


class SparseTransformerDualEncoder(nn.Module):
    """Sparse Transformer Model for Matching (dual encoding) tasks."""

    attention_patterns: Any
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

    @nn.compact
    def __call__(self, inputs1, inputs2, *, inputs1_positions=None, inputs2_positions=None,
                 inputs1_segmentation=None, inputs2_segmentation=None, train: bool=False):
        encoder_module_kwargs = {"attention_patterns": self.attention_patterns}
        encoded = generic.GenericDualEncoder(
            encoder_module=SparseTransformerEncoder,
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


class SparseTransformerDecoder(nn.Module):
    """Sparse Transformer Decoder."""

    attention_patterns: Any
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

    @nn.compact
    def __call__(self, inputs, *, train: bool=False):
        block_module_kwargs = {
            "attention_module_kwargs": {"attention_patterns": self.attention_patterns}
        }
        logits = generic.GenericDecoder(
            block_module=SparseTransformerBlock,
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

