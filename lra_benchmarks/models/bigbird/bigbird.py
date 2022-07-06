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
"""Transformer using BigBird (https://arxiv.org/abs/2007.14062)."""
from functools import partial
from typing import Any, Optional

from flax import linen as nn
import jax.numpy as jnp
import jax.nn as jnn

from lra_benchmarks.models.bigbird import bigbird_attention
from lra_benchmarks.models.layers import common_layers
from lra_benchmarks.models.generic import generic

_DEFAULT_BLOCK_SIZE = 64


BigBirdBlock = partial(generic.GenericBlock,
                       attention_module=bigbird_attention.BigBirdSelfAttention)

block_size=_DEFAULT_BLOCK_SIZE
connectivity_seed=None

block_size=block_size
connectivity_seed=connectivity_seed


class BigBirdEncoder(nn.Module):
    """BigBird Model Encoder."""

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
    classifier_pool: Any='CLS'
    num_classes: int=10
    tied_weights: bool=False
    block_size: int=_DEFAULT_BLOCK_SIZE

    def setup(self):
        if self.classifier and self.classifier_pool == 'CLS':
            self._max_len = self.max_len + 1
        else:
            self._max_len = self.max_len

    @nn.compact
    def __call__(self, inputs, *, inputs_positions=None, inputs_segmentation=None, train=True,
                 block_kwargs: Optional[dict[str, Any]]=None):
        """Applies BigBird transformer model on the inputs.

        Args:
            inputs: input data
            vocab_size: size of the vocabulary
            inputs_positions: input subsequence positions for packed examples.
            inputs_segmentation: input segmentation info for packed examples.
            shared_embedding: a shared embedding layer to use.
            use_bfloat16: bool: whether use bfloat16.
            emb_dim: dimension of embedding
            num_heads: number of heads
            dtype: the dtype of the computation (default: float32)
            num_layers: number of layers
            qkv_dim: dimension of the query/key/value
            mlp_dim: dimension of the mlp on top of attention block
            max_len: maximum length.
            train: if it is training,
            dropout_rate: dropout rate
            attention_dropout_rate: dropout rate for attention weights
            learn_pos_emb: boolean, if learn the positional embedding or use the
                sinusoidal positional embedding.
            classifier: boolean, for classification mode (output N-class logits)
            classifier_pool: str, supports "MEAN", "MAX" pooling.
            num_classes: int, number of classification classes.
            block_size: Size of attention blocks.

        Returns:
            output of a transformer encoder or logits if classifier_mode is true.
        """
        if self.tied_weights:
            raise NotImplementedError

        assert inputs.ndim == 2  # (batch, len)

        # Padding Masks
        src_padding_mask = (inputs > 0)[..., None]
        src_padding_mask = jnp.reshape(src_padding_mask, inputs.shape)  # (batch, len)

        # Input Embedding
        if self.shared_embedding is None:
            input_embed = nn.Embed(
                    num_embeddings=self.vocab_size,
                    features=self.emb_dim,
                    embedding_init=jnn.initializers.normal(stddev=1.0))
        else:
            input_embed = self.shared_embedding
        x = inputs.astype('int32')
        x = input_embed(x)

        if self.classifier and self.classifier_pool == 'CLS':
            cls = self.param('cls', jnn.initializers.zeros, (1, 1, self.emb_dim))
            cls = jnp.tile(cls, [x.shape[0], 1, 1])
            x = jnp.concatenate([cls, x], axis=1)
            src_padding_mask = jnp.concatenate(
                    [src_padding_mask[:, :1], src_padding_mask], axis=1)

        pe_init = jnn.initializers.normal(stddev=0.02) if self.learn_pos_emb else None
        x = common_layers.AddPositionEmbs(
                inputs_positions=inputs_positions,
                posemb_init=pe_init,
                max_len=self._max_len,
                name='posembed_input')(x)
        x = nn.Dropout(rate=self.dropout_rate)(x, deterministic=not train)

        if self.use_bfloat16:
            x = x.astype(jnp.bfloat16)
            dtype = jnp.bfloat16
        else:
            dtype = jnp.float32

        # Input Encoder
        for lyr in range(self.num_layers):
            attention_module_kwargs = {"block_size": self.block_size,
                                       "connectivity_seed": lyr}
            x = BigBirdBlock(
                    qkv_dim=self.qkv_dim,
                    mlp_dim=self.mlp_dim,
                    num_heads=self.num_heads,
                    dtype=dtype,
                    dropout_rate=self.dropout_rate,
                    attention_dropout_rate=self.attention_dropout_rate,
                    max_len=self._max_len,
                    name=f'encoderblock_{lyr}',
                    attention_module_kwargs=attention_module_kwargs,
            )(x, inputs_segmentation=inputs_segmentation, padding_mask=src_padding_mask,
              deterministic=not train)
        encoded = nn.LayerNorm(dtype=dtype, name='encoder_norm')(x)

        if self.classifier:
            encoded = common_layers.classifier_head(
                    encoded, self.num_classes, self.mlp_dim, pooling_mode=self.classifier_pool)
        return encoded

BigBirdDualEncoder = partial(generic.GenericDualEncoder,
                             encoder_module=BigBirdEncoder)

BigBirdDecoder = partial(generic.GenericDecoder, block_module=BigBirdBlock)

