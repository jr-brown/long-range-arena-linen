"""Sinkhorn Attention Transformer models."""
from functools import partial
from typing import Any

from flax import linen as nn
import jax.numpy as jnp

from lra_benchmarks.models.sinkhorn_transformer.sinkhorn_attention import SinkhornSelfAttention
from lra_benchmarks.models.generic import generic
from lra_benchmarks.models.generic.module_collection import ModuleCollection


class SinkhornEncoder(nn.Module):
    """Sinkhorn Transformer Encoder."""

    block: Any
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
    classifier_pool: Any='MEAN'
    num_classes: int=10

    def setup(self):
        if self.classifier and self.classifier_pool == 'CLS':
            self._max_len = self.max_len + 1
        else:
            self._max_len = self.max_len

    @nn.compact
    def __call__(self, inputs, *, inputs_positions=None, inputs_segmentation=None, train=True):
        def custom_classifier_func(encoded):
            if self.classifier_pool == 'MEAN':
                encoded = jnp.mean(encoded, axis=1)
                return nn.Dense(self.num_classes, name='logits')(encoded)
            else:
                # TODO(yitay): Add other pooling methods.
                raise ValueError(f'{self.classifier_pool} Pooling method not supported yet.')

        x = generic.GenericEncoder(
            block_module=self.block,
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
            custom_classifier_func=custom_classifier_func,
        )(inputs, inputs_positions=inputs_positions, inputs_segmentation=inputs_segmentation,
          train=train)

        return x


def get_modules(block_size: int=20, max_num_blocks: int=25, sort_activation: str="softmax"
                ) -> ModuleCollection:
    attn = partial(SinkhornSelfAttention, block_size=block_size, max_num_blocks=max_num_blocks,
                   sort_activation=sort_activation)
    block = partial(generic.GenericBlock, attention_module=attn)

    return ModuleCollection(attn, block=block, encoder=partial(SinkhornEncoder, block=block))

