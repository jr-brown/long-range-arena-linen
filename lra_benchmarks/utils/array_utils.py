import flax.linen as nn
import jax.numpy as jnp
from jax import lax


def combine_masks_into_bias(masks: list, *, dtype=None):
    if masks:
        attention_mask = nn.combine_masks(*masks)

        # attention mask in the form of attention bias
        return lax.select(attention_mask > 0,
                          jnp.full(attention_mask.shape, 0.).astype(dtype),
                          jnp.full(attention_mask.shape, -1e10).astype(dtype))
    else:
        return None


def make_block_attention_mask(*, seq_shape, bs, num_query_blocks, block_size, num_heads, dtype=None,
                              causal_mask=False, padding_mask=None, key_padding_mask=None,
                              segmentation=None, key_segmentation=None, use_attention_bias=False):
    # create attention masks
    mask_components = []

    if causal_mask:
        mask_components.append(nn.make_causal_mask(jnp.zeros(seq_shape)))

    if padding_mask is not None:
        padding_mask = jnp.reshape(padding_mask, (bs * num_query_blocks, block_size))

        if key_padding_mask is None:
            key_padding_mask = padding_mask
        else:
            key_padding_mask = jnp.reshape(key_padding_mask, (bs*num_query_blocks, block_size))

        pad_mask = nn.make_attention_mask(padding_mask, key_padding_mask)
        # To get into shape (bs, num_query_blocks, num_heads, block_size, block_size)
        pad_mask = jnp.reshape(jnp.repeat(pad_mask, num_heads, axis=0),
                               (bs, num_query_blocks, num_heads, block_size, block_size))
        mask_components.append(pad_mask)

    if segmentation is not None:
        if key_segmentation is None:
            key_segmentation = segmentation
        segmentation_mask = nn.make_attention_mask(segmentation, key_segmentation,
                                                   pairwise_fn=jnp.equal)
        segmentation_mask = jnp.reshape(jnp.repeat(segmentation_mask, num_heads, axis=1),
                                        (bs, num_query_blocks, num_heads, block_size, block_size))
        mask_components.append(segmentation_mask)

    if mask_components:
        if use_attention_bias:
            attention_mask = combine_masks_into_bias(mask_components, dtype=dtype)
        else:
            attention_mask = nn.combine_masks(*mask_components)
    else:
        attention_mask = None

    return mask_components, attention_mask


def make_attention_mask(*, seq_shape, dtype=None, causal_mask=False, padding_mask=None,
                        key_padding_mask=None, segmentation=None, key_segmentation=None,
                        use_attention_bias=False, extra_masks=None):
    mask_components = [] if extra_masks is None else extra_masks

    if causal_mask:
        mask_components.append(nn.make_causal_mask(jnp.zeros(seq_shape)))

    if padding_mask is not None:
        if key_padding_mask is None:
            key_padding_mask = padding_mask
        padding_mask = nn.make_attention_mask(padding_mask, key_padding_mask)
        mask_components.append(padding_mask)

    if segmentation is not None:
        if key_segmentation is None:
            key_segmentation = segmentation
        segmentation_mask = nn.make_attention_mask(segmentation, key_segmentation,
                                                   pairwise_fn=jnp.equal)
        mask_components.append(segmentation_mask)

    if mask_components:
        if use_attention_bias:
            attention_mask = combine_masks_into_bias(mask_components, dtype=dtype)
        else:
            attention_mask = nn.combine_masks(*mask_components)
    else:
        attention_mask = None

    return mask_components, attention_mask

