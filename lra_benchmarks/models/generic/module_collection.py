from typing import Optional, Union, TypeVar, Type
from functools import partial

from flax.linen import Module

from lra_benchmarks.models.generic import generic


A = TypeVar('A')

PossibleModule = Optional[Union[Type[Module], partial[Module], Type[A], partial[A]]]


class ModuleCollection:
    def __init__(self,
                 block: PossibleModule[generic.GenericBlock]=None,
                 encoder: PossibleModule[generic.GenericEncoder]=None,
                 dual_encoder: PossibleModule[generic.GenericDualEncoder]=None,
                 decoder: PossibleModule[generic.GenericDecoder]=None):

        # Block is always required, the others are inferred if not explicitly provided
        self.block = block

        if encoder is None:
            self.encoder = partial(generic.GenericEncoder, block_module=self.block)
        else:
            self.encoder = encoder

        if dual_encoder is None:
            self.dual_encoder = partial(generic.GenericDualEncoder, encoder_module=self.encoder)
        else:
            self.dual_encoder = dual_encoder

        if decoder is None:
            self.decoder = partial(generic.GenericDecoder, block_module=self.block)
        else:
            self.decoder = decoder


    def __getitem__(self, key: str):
        module_dict = {
            "block": self.block,
            "encoder": self.encoder,
            "dual_encoder": self.dual_encoder,
            "decoder": self.decoder
        }
        return module_dict[key]
