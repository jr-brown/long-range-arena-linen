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
"""Sinkhorn Attention Transformer models."""
from functools import partial
from typing import Any

from flax import linen as nn
import jax.numpy as jnp

from lra_benchmarks.models.sinkhorn_transformer import sinkhorn_attention
from lra_benchmarks.models.generic import generic


SinkhornTransformerBlock = partial(generic.GenericBlock,
                                   attention_module=sinkhorn_attention.SinkhornSelfAttention)


class SinkhornTransformerEncoder(nn.Module):
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
    block_size: int=50
    learn_pos_emb: bool=False
    classifier: bool=False
    classifier_pool: Any='MEAN'
    num_classes: int=10

    def setup(self):
        if self.classifier and self.classifier_pool == 'CLS':
            self._max_len = self.max_len + 1
        else:
            self._max_len = self.max_len

    @nn.compact
    def __call__(self, inputs, *, inputs_positions=None, inputs_segmentation=None, train=True):
        block_module_kwargs={"attention_module_kwargs" : {"block_size": self.block_size}}

        def custom_classifier_func(encoded):
            if self.classifier_pool == 'MEAN':
                encoded = jnp.mean(encoded, axis=1)
                return nn.Dense(self.num_classes, name='logits')(encoded)
            else:
                # TODO(yitay): Add other pooling methods.
                raise ValueError(f'{self.classifier_pool} Pooling method not supported yet.')

        x = generic.GenericEncoder(
            block_module=SinkhornTransformerBlock,
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
            block_module_kwargs=block_module_kwargs,
            custom_classifier_func=custom_classifier_func,
        )(inputs, inputs_positions=inputs_positions, inputs_segmentation=inputs_segmentation,
          train=train)

        return x

SinkhornTransformerDualEncoder = partial(generic.GenericDualEncoder,
                                         encoder_module=SinkhornTransformerEncoder)
SinkhornTransformerDecoder = partial(generic.GenericDecoder, block_module=SinkhornTransformerBlock)

