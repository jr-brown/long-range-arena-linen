from typing import Any, Optional

from flax import linen as nn
import jax.numpy as jnp
import jax.nn as jnn

from lra_benchmarks.models.layers import common_layers


class GenericBlock(nn.Module):
    """Generic Layer"""

    attention_module: nn.Module
    qkv_dim: Any
    mlp_dim: Any
    num_heads: Any
    dtype: Any=jnp.float32
    dropout_rate: Any=0.1
    attention_dropout_rate: Any=0.1
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
                **attention_module_kwargs
        )(x, segmentation=inputs_segmentation,causal_mask=causal_mask, padding_mask=padding_mask,
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


class GenericDualEncoder(nn.Module):
    """Transformer Model for Matching (dual encoding) tasks."""

    encoder_module: nn.Module
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
