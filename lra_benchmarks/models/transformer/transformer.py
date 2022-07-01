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
"""Transformer model."""
from lra_benchmarks.models.layers import common_layers
from typing import Any

import flax.linen as nn
import jax.numpy as jnp
import jax.nn as jnn


class TransformerBlock(nn.Module):
    """Transformer layer (https://openreview.net/forum?id=H1e5GJBtDr)."""

    qkv_dim: Any
    mlp_dim: Any
    num_heads: Any
    dtype: Any=jnp.float32
    causal_mask: Any=False
    padding_mask: Any=None
    dropout_rate: Any=0.1
    attention_dropout_rate: Any=0.1
    decode: bool=False

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
            padding_mask: bool array, mask padding tokens
            dropout_rate: dropout rate
            attention_dropout_rate: dropout rate for attention weights
            decode: bool, whether or not we're decoding (originally cache)
            deterministic: bool, deterministic or not (to apply dropout)

        Returns:
            output after transformer block.

        """

        # Attention block.
        assert inputs.ndim == 3
        x = nn.LayerNorm()(inputs)

        mask = nn.make_attention_mask(padding_mask, padding_mask)

        if causal_mask:
            mask = nn.combine_masks(mask, nn.make_causal_mask(x))

        x = nn.SelfAttention(
                num_heads=self.num_heads,
                dtype=self.dtype,
                qkv_features=self.qkv_dim,
                kernel_init=jnn.initializers.xavier_uniform(),
                bias_init=jnn.initializers.normal(stddev=1e-6),
                use_bias=False,
                broadcast_dropout=False,
                dropout_rate=self.attention_dropout_rate,
                decode=self.decode
        )(x, deterministic=deterministic, mask=mask)
        x = nn.Dropout(rate=self.dropout_rate)(x, deterministic=deterministic)
        x = x + inputs

        # MLP block.
        y = nn.LayerNorm()(x)
        y = common_layers.MlpBlock(
            mlp_dim=self.mlp_dim, dtype=self.dtype, dropout_rate=self.dropout_rate
        )(y, deterministic=deterministic)

        return x + y


class TransformerEncoder(nn.Module):
    """Transformer Model Encoder."""

    vocab_size: Any
    shared_embedding: Any=None
    use_bfloat16: Any=False
    emb_dim: Any=512
    num_heads: Any=8
    dtype: Any=jnp.float32
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
    tied_weights: Any=False

    def setup(self):
        if self.classifier and self.classifier_pool == 'CLS':
            self._max_len = self.max_len + 1
        else:
            self._max_len = self.max_len

    @nn.compact
    def __call__(self, inputs, inputs_positions=None, train=True):
        """Applies Transformer model on the inputs.

        Args:
            inputs: input data
            vocab_size: size of the vocabulary
            inputs_positions: input subsequence positions for packed examples.
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
            tied_weights: bool, to tie weights or not.

        Returns:
            output of a transformer encoder or logits if classifier_mode is true.
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

        # Input Encoder
        if self.tied_weights:
            encoder = TransformerBlock(
                    qkv_dim=self.qkv_dim,
                    mlp_dim=self.mlp_dim,
                    num_heads=self.num_heads,
                    dtype=dtype,
                    dropout_rate=self.dropout_rate,
                    attention_dropout_rate=self.attention_dropout_rate,
                    name='encoderblock')
            for _ in range(self.num_layers):
                x = encoder(x, padding_mask=src_padding_mask, deterministic=not train)
        else:
            for lyr in range(self.num_layers):
                x = TransformerBlock(
                        qkv_dim=self.qkv_dim,
                        mlp_dim=self.mlp_dim,
                        num_heads=self.num_heads,
                        dtype=dtype,
                        dropout_rate=self.dropout_rate,
                        attention_dropout_rate=self.attention_dropout_rate,
                        name=f'encoderblock_{lyr}'
                )(x, padding_mask=src_padding_mask, deterministic=not train)

        encoded = nn.LayerNorm(dtype=dtype, name='encoder_norm')(x)

        if self.classifier:
            encoded = common_layers.classifier_head(
                    encoded, self.num_classes, self.mlp_dim, pooling_mode=self.classifier_pool)
        return encoded


class TransformerDualEncoder(nn.Module):
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

        encoder = TransformerEncoder(
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
