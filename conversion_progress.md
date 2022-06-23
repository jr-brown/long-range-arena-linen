
# Linen conversion progress

Using https://flax.readthedocs.io/en/latest/howtos/linen_upgrade_guide.html#defining-simple-modules as a guide.

### Quick key
* N | Not implemented
* PX | Partially implemented, only for certain tasks / transformers, denoted by the X
* D | Fully implemented except docs & comments
* F | Fully implemented including docs & comments

When something is not applicable it can be written as implemented to reflect the fact it has been checked if it needed to change, even if no changes actually occoured.

### Partial implementation groups
* P1 - layers.common_layers


## Defining Simple Modules
1. D
2. P1
3. P1
4. P1
5. P1

## Using Modules inside other Modules
1. P1
2. P1

## Sharing submodules and defining multiple methods
1. P1

## Module.partial inside other modules
1. D

## Top-level training code patterns
1. P1
2. P1
3. P1
4. P1
5. P1

## Non-trainable variables (“state”): Use within Modules
1. P1

## Non-trainable variables (“state”): Top-level training code patterns
1. P1
2. P1
3. P1
4. P1

## Loading pre-Linen checkpoints
1. P1
2. P1

## Lifted Transforms
1. P1

## Initialisers to use jax.nn
1. P1

