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
from typing import Any, Optional

from flax import linen as nn
import jax.numpy as jnp

from lra_benchmarks.models.sparse_transformer.sparse_attention import SparseSelfAttention
from lra_benchmarks.models.generic import generic
from lra_benchmarks.models.generic.module_collection import ModuleCollection


def get_modules(attention_pattern_args: Optional[list[tuple[str, dict[str, Any]]]]=None
                ) -> ModuleCollection:
    attn = partial(SparseSelfAttention, attention_pattern_args=attention_pattern_args)
    block = partial(generic.GenericBlock, attention_module=attn)

    class encoder(nn.Module):
        """Local Transformer Encoder."""

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
                    "use_cls_token": use_cls_token,
                }
            }

            x = generic.GenericEncoder(
                block_module=block,
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

    return ModuleCollection(attn, block=block, encoder=encoder)

