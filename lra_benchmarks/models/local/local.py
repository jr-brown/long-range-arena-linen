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
from functools import partial
from typing import Any

from flax import linen as nn
import jax.numpy as jnp
import jax.nn as jnn

from lra_benchmarks.models.layers import common_layers
from lra_benchmarks.models.local import local_attention


class LocalTransformerBlock(nn.Module):
    """Transformer layer (https://openreview.net/forum?id=H1e5GJBtDr)."""

    qkv_dim: Any
    mlp_dim: Any
    num_heads: Any
    dtype: Any=jnp.float32
    dropout_rate: Any=0.1
    attention_dropout_rate: Any=0.1
    block_size: Any=50
    max_len: int=512

    @nn.compact
    def __call__(self, inputs, *, causal_mask: bool=False, padding_mask=None,
                 deterministic: bool=False):
        """Applies TransformerBlock module.

        Args:
            inputs: input data
            qkv_dim: dimension of the query/key/value
            mlp_dim: dimension of the mlp on top of attention block
            num_heads: number of heads
            dtype: the dtype of the computation (default: float32).
            causal_mask: bool, mask future or not
            padding_mask: bool, mask padding tokens
            dropout_rate: dropout rate
            attention_dropout_rate: dropout rate for attention weights
            deterministic: bool, deterministic or not (to apply dropout)
            block_size: int, block size.

        Returns:
            output after transformer block.

        """

        # Attention block.
        assert inputs.ndim == 3
        x = nn.LayerNorm()(inputs)
        x = local_attention.LocalSelfAttention(
                num_heads=self.num_heads,
                dtype=self.dtype,
                qkv_features=self.qkv_dim,
                kernel_init=jnn.initializers.xavier_uniform(),
                bias_init=jnn.initializers.normal(stddev=1e-6),
                bias=False,
                broadcast_dropout=False,
                dropout_rate=self.attention_dropout_rate,
                block_size=self.block_size,
                max_len=self.max_len
        )(x, causal_mask=causal_mask, padding_mask=padding_mask,
          deterministic=deterministic)
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
    def __call__(self, inputs, *, inputs_positions=None, train=True):
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
            x = LocalTransformerBlock(
                    qkv_dim=self.qkv_dim,
                    mlp_dim=self.mlp_dim,
                    num_heads=self.num_heads,
                    dtype=dtype,
                    dropout_rate=self.dropout_rate,
                    attention_dropout_rate=self.attention_dropout_rate,
                    name=f'encoderblock_{lyr}',
                    block_size=self.block_size,
                    max_len=self._max_len
            )(x, padding_mask=src_padding_mask, deterministic=not train)
        encoded = nn.LayerNorm(dtype=dtype, name='encoder_norm')(x)

        if self.classifier:
            encoded = common_layers.classifier_head(
                    encoded, self.num_classes, self.mlp_dim, pooling_mode=self.classifier_pool)
        return encoded


class LocalTransformerDualEncoder(nn.Module):
    """Transformer Model for Matching (dual encoding) tasks."""

    vocab_size: Any=None
    use_bfloat16: Any=False
    emb_dim: Any=512
    num_heads: Any=8
    num_layers: Any=6
    qkv_dim: Any=512
    mlp_dim: Any=2048
    max_len: Any=2048
    dropout_rate: Any=0.1
    attention_dropout_rate: Any=0.1
    classifier: Any=True
    classifier_pool: Any='CLS'
    num_classes: Any=2
    interaction: Any=None

    @nn.compact
    def __call__(self, inputs1, inputs2, inputs1_positions=None, inputs2_positions=None,
                 train: bool=False):
        """Applies Transformer model on text similarity.

        A deliberate choice to distinguish this from NLI because
        we may want to do different things to the model later. Dual Encoding
        mode enforces that we do not do cross attention between pairs.

        Args:
            inputs1: input data.
            inputs2: target data.
            vocab_size: size of the input vocabulary.
            inputs1_positions: input subsequence positions for packed examples.
            inputs2_positions: target subsequence positions for packed examples.
            use_bfloat16: bool: whether use bfloat16.
            emb_dim: dimension of embedding.
            num_heads: number of heads.
            num_layers: number of layers.
            qkv_dim: dimension of the query/key/value.
            mlp_dim: dimension of the mlp on top of attention block.
            max_len: maximum length.
            train: whether it is training.
            dropout_rate: dropout rate.
            attention_dropout_rate: dropout rate for attention weights.
            classifier: boolean, to use classifier.
            classifier_pool: str, supports "MEAN", "MAX" pooling.
            num_classes: int, number of classification classes.
            interaction: str, supports "NLI"

        Returns:
            output of a transformer decoder.
        """

        encoder = LocalTransformerEncoder(
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
                name='encoder')
        inputs1_encoded = encoder(
                inputs=inputs1,
                inputs_positions=inputs1_positions,
                train=train)
        inputs2_encoded = encoder(
                inputs=inputs2,
                inputs_positions=inputs2_positions,
                train=train)

        encoded = common_layers.classifier_head_dual(
                inputs1_encoded,
                inputs2_encoded,
                self.num_classes,
                self.mlp_dim,
                pooling_mode=self.classifier_pool,
                interaction=self.interaction)

        return encoded


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
            raise NotImplementedError
            # x = common_layers.shift_right(x)
        x = x.astype('int32')
        x = common_layers.Embed(num_embeddings=self.vocab_size, features=self.emb_dim, name='embed')(x)
        x = common_layers.AddPositionEmbs(
                max_len=self.max_len,
                posemb_init=common_layers.sinusoidal_init(max_len=self.max_len))(x)
        x = nn.Dropout(rate=self.dropout_rate)(x, deterministic=not train)
        for _ in range(self.num_layers):
            x = LocalTransformerBlock(
                    qkv_dim=self.qkv_dim,
                    mlp_dim=self.mlp_dim,
                    num_heads=self.num_heads,
                    dropout_rate=self.dropout_rate,
                    attention_dropout_rate=self.attention_dropout_rate,
                    block_size=self.block_size
            )(x, causal_mask=True, padding_mask=padding_mask, deterministic=not train)
        x = nn.LayerNorm(x)
        logits = nn.Dense(
                self.vocab_size,
                kernel_init=jnn.initializers.xavier_uniform(),
                bias_init=jnn.initializers.normal(stddev=1e-6))(x)
        return logits
