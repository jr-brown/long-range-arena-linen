from typing import Any, Optional

from flax import linen as nn
import jax.numpy as jnp
import jax.nn as jnn

from lra_benchmarks.models.layers import common_layers


class GenericBlock(nn.Module):
    """Generic Layer"""

    attention_module: nn.Module
    qkv_dim: int
    mlp_dim: int
    num_heads: int
    dtype: Any=jnp.float32
    dropout_rate: float=0.1
    attention_dropout_rate: float=0.1
    max_len: int=512
    attention_module_kwargs: Optional[dict[str, Any]]=None

    @nn.compact
    def __call__(self, inputs, *, inputs_segmentation=None, causal_mask: bool=False,
                 padding_mask=None, deterministic: bool=False,
                 attention_kwargs: Optional[dict[str, Any]]=None):
        """Applies GenericBlock module.

        Args:
            TODO

        Returns:
            Output after the block

        """

        if attention_kwargs is None:
            attention_kwargs = {}

        if self.attention_module_kwargs is None:
            attention_module_kwargs = {}
        else:
            attention_module_kwargs = self.attention_module_kwargs

        # Attention block.
        assert inputs.ndim == 3
        x = nn.LayerNorm()(inputs)
        x = self.attention_module(
                num_heads=self.num_heads,
                dtype=self.dtype,
                qkv_features=self.qkv_dim,
                kernel_init=jnn.initializers.xavier_uniform(),
                bias_init=jnn.initializers.normal(stddev=1e-6),
                bias=False,
                broadcast_dropout=False,
                dropout_rate=self.attention_dropout_rate,
                max_len=self.max_len,
                **attention_module_kwargs
        )(x, segmentation=inputs_segmentation, causal_mask=causal_mask, padding_mask=padding_mask,
          deterministic=deterministic, **attention_kwargs)
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


class GenericEncoder(nn.Module):
    """Transformer Model Encoder."""

    block_module: nn.Module
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
    block_module_kwargs: Optional[dict[str, Any]]=None
    custom_classifier_func: Any=None

    def setup(self):
        if self.classifier and self.classifier_pool == 'CLS':
            self._max_len = self.max_len + 1
        else:
            self._max_len = self.max_len

    @nn.compact
    def __call__(self, inputs, *, inputs_positions=None, inputs_segmentation=None, train=True,
                 block_kwargs: Optional[dict[str, Any]]=None):
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
        if self.block_module_kwargs is None:
            block_module_kwargs = {}
        else:
            block_module_kwargs = self.block_module_kwargs

        if block_kwargs is None:
            block_kwargs = {}

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
            encoder = self.block_module(
                    qkv_dim=self.qkv_dim,
                    mlp_dim=self.mlp_dim,
                    num_heads=self.num_heads,
                    dtype=dtype,
                    dropout_rate=self.dropout_rate,
                    attention_dropout_rate=self.attention_dropout_rate,
                    max_len=self._max_len,
                    name='encoderblock',
                    **block_module_kwargs)
            for _ in range(self.num_layers):
                x = encoder(x, inputs_segmentation=inputs_segmentation,
                            padding_mask=src_padding_mask, deterministic=not train, **block_kwargs)
        else:
            for lyr in range(self.num_layers):
                x = self.block_module(
                        qkv_dim=self.qkv_dim,
                        mlp_dim=self.mlp_dim,
                        num_heads=self.num_heads,
                        dtype=dtype,
                        dropout_rate=self.dropout_rate,
                        attention_dropout_rate=self.attention_dropout_rate,
                        max_len=self._max_len,
                        name=f'encoderblock_{lyr}',
                        **block_module_kwargs
                )(x, inputs_segmentation=inputs_segmentation, padding_mask=src_padding_mask,
                  deterministic=not train, **block_kwargs)

        encoded = nn.LayerNorm(dtype=dtype, name='encoder_norm')(x)

        if self.classifier:
            if self.custom_classifier_func is None:
                encoded = common_layers.classifier_head(encoded, self.num_classes, self.mlp_dim,
                                                        pooling_mode=self.classifier_pool)
            else:
                encoded = self.custom_classifier_func(encoded)
        return encoded


class GenericDualEncoder(nn.Module):
    """Transformer Model for Matching (dual encoding) tasks."""

    encoder_module: nn.Module
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
    encoder_module_kwargs: Optional[dict[str, Any]]=None

    @nn.compact
    def __call__(self, inputs1, inputs2, *, inputs1_positions=None, inputs2_positions=None,
                 inputs1_segmentation=None, inputs2_segmentation=None, train: bool=False):
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

        if self.encoder_module_kwargs is None:
            encoder_module_kwargs = {}
        else:
            encoder_module_kwargs = self.encoder_module_kwargs

        encoder = self.encoder_module(
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
                name='encoder',
                **encoder_module_kwargs)
        inputs1_encoded = encoder(
                inputs=inputs1,
                inputs_positions=inputs1_positions,
                inputs_segmentation=inputs1_segmentation,
                train=train)
        inputs2_encoded = encoder(
                inputs=inputs2,
                inputs_positions=inputs2_positions,
                inputs_segmentation=inputs2_segmentation,
                train=train)

        encoded = common_layers.classifier_head_dual(
                inputs1_encoded,
                inputs2_encoded,
                self.num_classes,
                self.mlp_dim,
                pooling_mode=self.classifier_pool,
                interaction=self.interaction)

        return encoded


class GenericDecoder(nn.Module):
    """Local Transformer Decoder."""

    block_module: nn.Module
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
    block_module_kwargs: Optional[dict[str, Any]]=None

    @nn.compact
    def __call__(self, inputs, *, train: bool=False,
                 block_kwargs: Optional[dict[str, Any]]=None):
        """Applies Transformer model on the inputs.

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

        Returns:
            output of a transformer decoder.
        """

        if self.block_module_kwargs is None:
            block_module_kwargs = {}
        else:
            block_module_kwargs = self.block_module_kwargs

        if block_kwargs is None:
            block_kwargs = {}

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

        for _ in range(self.num_layers):
            x = self.block_module(
                    qkv_dim=self.qkv_dim,
                    mlp_dim=self.mlp_dim,
                    num_heads=self.num_heads,
                    dropout_rate=self.dropout_rate,
                    attention_dropout_rate=self.attention_dropout_rate,
                    **block_module_kwargs
            )(x, causal_mask=True, padding_mask=padding_mask, deterministic=not train,
              **block_kwargs)
        x = nn.LayerNorm()(x)
        logits = nn.Dense(
                self.vocab_size,
                kernel_init=jnn.initializers.xavier_uniform(),
                bias_init=jnn.initializers.normal(stddev=1e-6))(x)
        return logits

