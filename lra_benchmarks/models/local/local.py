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
"""Local Attention Transformer models."""
from typing import Any
from functools import partial

from flax import linen as nn
import jax.numpy as jnp
import jax.nn as jnn

from lra_benchmarks.models.layers import common_layers
from lra_benchmarks.models.local import local_attention
from lra_benchmarks.models.generic import generic


class LocalTransformerEncoder(nn.Module):
    """Local Transformer Encoder."""

    vocab_size: Any
    shared_embedding: Any=None
    use_bfloat16: Any=False
    emb_dim: Any=512
    num_heads: Any=8
    num_layers: Any=6
    qkv_dim: Any=512
    mlp_dim: Any=2048
    max_len: Any=512
    dropout_rate: Any=0.1
    attention_dropout_rate: Any=0.1
    block_size: Any=50
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
        """Applies Local Transformer model on the inputs.

        Args:
            inputs: input data
            vocab_size: size of the vocabulary
            inputs_positions: input subsequence positions for packed examples.
            shared_embedding: a shared embedding layer to use.
            use_bfloat16: bool: whether use bfloat16.
            emb_dim: dimension of embedding
            num_heads: number of heads
            num_layers: number of layers
            qkv_dim: dimension of the query/key/value
            mlp_dim: dimension of the mlp on top of attention block
            max_len: maximum length.
            train: if it is training,
            dropout_rate: dropout rate
            attention_dropout_rate: dropout rate for attention weights
            block_size: int, block size.
            learn_pos_emb: boolean, if learn the positional embedding or use the
                sinusoidal positional embedding.
            classifier: boolean, for classification mode (output N-class logits)
            classifier_pool: str, supports "MEAN", "MAX" pooling.
            num_classes: int, number of classification classes.

        Returns:
            output of a transformer encoder.
        """
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

        attention_module_kwargs = {
            "block_size": self.block_size,
            "max_len": self._max_len
        }

        # Input Encoder
        for lyr in range(self.num_layers):
            x = generic.GenericBlock(
                    attention_module=local_attention.LocalSelfAttention,
                    qkv_dim=self.qkv_dim,
                    mlp_dim=self.mlp_dim,
                    num_heads=self.num_heads,
                    dtype=dtype,
                    dropout_rate=self.dropout_rate,
                    attention_dropout_rate=self.attention_dropout_rate,
                    name=f'encoderblock_{lyr}',
                    attention_module_kwargs=attention_module_kwargs
            )(x, inputs_segmentation=inputs_segmentation, padding_mask=src_padding_mask,
              deterministic=not train)
        encoded = nn.LayerNorm(dtype=dtype, name='encoder_norm')(x)

        if self.classifier:
            encoded = common_layers.classifier_head(
                    encoded, self.num_classes, self.mlp_dim, pooling_mode=self.classifier_pool)
        return encoded


LocalTransformerDualEncoder = partial(generic.GenericDualEncoder,
                                      encoder_module=LocalTransformerEncoder)


class LocalTransformerDecoder(nn.Module):
    """Local Transformer Decoder."""

    vocab_size: Any
    emb_dim: Any=512
    num_heads: Any=8
    num_layers: Any=6
    qkv_dim: Any=512
    mlp_dim: Any=2048
    max_len: Any=2048
    shift: Any=True
    dropout_rate: Any=0.1
    attention_dropout_rate: Any=0.1
    block_size: Any=50

    @nn.compact
    def __call__(self, inputs, train: bool=False):
        """Applies Local Attention model on the inputs.

        Args:
            inputs: input data
            vocab_size: size of the vocabulary
            emb_dim: dimension of embedding
            num_heads: number of heads
            num_layers: number of layers
            qkv_dim: dimension of the query/key/value
            mlp_dim: dimension of the mlp on top of attention block
            max_len: maximum length.
            train: bool: if model is training.
            shift: bool: if we right-shift input - this is only disabled for
                fast, looped single-token autoregressive decoding.
            dropout_rate: dropout rate
            attention_dropout_rate: dropout rate for attention weights
            block_size: int, block size

        Returns:
            output of a transformer decoder.
        """
        padding_mask = jnp.where(inputs > 0, 1, 0).astype(jnp.float32)[..., None]
        assert inputs.ndim == 2  # (batch, len)
        x = inputs
        if self.shift:
            x = common_layers.shift_right(x)
        x = x.astype('int32')
        x = common_layers.Embed(num_embeddings=self.vocab_size, features=self.emb_dim,
                                name='embed')(x)
        x = common_layers.AddPositionEmbs(
                max_len=self.max_len,
                posemb_init=common_layers.sinusoidal_init(max_len=self.max_len))(x)
        x = nn.Dropout(rate=self.dropout_rate)(x, deterministic=not train)

        attention_module_kwargs = {"block_size": self.block_size}

        for _ in range(self.num_layers):
            x = generic.GenericBlock(
                    attention_module=local_attention.LocalSelfAttention,
                    qkv_dim=self.qkv_dim,
                    mlp_dim=self.mlp_dim,
                    num_heads=self.num_heads,
                    dropout_rate=self.dropout_rate,
                    attention_dropout_rate=self.attention_dropout_rate,
                    attention_module_kwargs=attention_module_kwargs
            )(x, causal_mask=True, padding_mask=padding_mask, deterministic=not train)
        x = nn.LayerNorm()(x)
        logits = nn.Dense(
                self.vocab_size,
                kernel_init=jnn.initializers.xavier_uniform(),
                bias_init=jnn.initializers.normal(stddev=1e-6))(x)
        return logits
