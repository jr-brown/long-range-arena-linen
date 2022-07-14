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
"""This contains utility functions for model training and evaluation."""
from functools import partial
from absl import logging
import itertools
import json
import os
import time

import jax
import jax.numpy as jnp
from jax import random
from jax.tree_util import tree_map

from flax import jax_utils
from flax import linen as nn
from flax.metrics import tensorboard
from flax.training import common_utils, train_state, checkpoints

import tensorflow.compat.v2 as tf
import optax
import numpy as onp

from lra_benchmarks.models.bigbird import bigbird
from lra_benchmarks.models.linear_transformer import linear_transformer
from lra_benchmarks.models.linformer import linformer
from lra_benchmarks.models.local import local
from lra_benchmarks.models.longformer import longformer
from lra_benchmarks.models.performer import performer
from lra_benchmarks.models.reformer import reformer
from lra_benchmarks.models.sinkhorn_transformer import sinkhorn_transformer
from lra_benchmarks.models.sparse_transformer import sparse_attention
from lra_benchmarks.models.sparse_transformer import sparse_transformer
from lra_benchmarks.models.synthesizer import synthesizer
from lra_benchmarks.models.transformer import transformer

from lra_benchmarks.utils.device_utils import shard
from lra_benchmarks.utils.misc_utils import r4

from extra_models import extra_models


def get_model(model_type, model_kwargs, init_rng, input_shapes, tx):
    """Create and initialize the model.

    Args:
        model_type: str; Type of Transformer model to create.
        create_model_fn: fn: Function that is used for creating the model.
        model_kwargs: keyword argument to the model.
        *create_model_args: positional argument to the create_model_args.

    Returns:
        Initialized model.
    """

    if model_type == 'sparse_transformer' or model_type == 'sparse_transformer_dual':
        model_kwargs['attention_patterns'] = [
            sparse_attention.Fixed1Pattern(block_size=50),
            sparse_attention.Fixed2Pattern(block_size=50, c=10)
        ]

    model_map = {
        "transformer": transformer.TransformerEncoder,
        "transformer_dual": transformer.TransformerDualEncoder,
        "local": local.LocalTransformerEncoder,
        "local_dual": local.LocalTransformerDualEncoder,
        "longformer": longformer.LongformerEncoder,
        "longformer_dual": longformer.LongformerDualEncoder,
        "reformer": reformer.ReformerEncoder,
        "reformer_dual": reformer.ReformerDualEncoder,
        "linformer": linformer.LinformerEncoder,
        "linformer_dual": linformer.LinformerDualEncoder,
        "sinkhorn": sinkhorn_transformer.SinkhornTransformerEncoder,
        "sinkhorn_dual": sinkhorn_transformer.SinkhornTransformerDualEncoder,
        "linear_transformer": linear_transformer.LinearTransformerEncoder,
        "linear_transformer_dual": linear_transformer.LinearTransformerDualEncoder,
        "bigbird": bigbird.BigBirdEncoder,
        "bigbird_dual": bigbird.BigBirdDualEncoder,
        "synthesizer": synthesizer.SynthesizerEncoder,
        "synthesizer_dual": synthesizer.SynthesizerDualEncoder,
        "sparse_transformer": sparse_transformer.SparseTransformerEncoder,
        "sparse_transformer_dual": sparse_transformer.SparseTransformerDualEncoder,
        "performer": performer.PerformerEncoder,
        "performer_dual": performer.PerformerDualEncoder,
    }

    model_map.update(extra_models)

    return create_train_state(model_map[model_type], model_kwargs, init_rng, input_shapes, tx)


def create_train_state(flax_module, model_kwargs, init_rng, input_shapes, tx
                       ) -> train_state.TrainState:
    """Creates and initializes the model."""

    @partial(jax.jit, backend='cpu')
    def _create_train_state(init_rng):
        module = flax_module(**model_kwargs)
        variables = module.init(init_rng, *[jnp.ones(s) for s in input_shapes], train=False)
        params = variables['params']
        return train_state.TrainState.create(apply_fn=module.apply, params=params, tx=tx)

    return _create_train_state(init_rng)


def create_learning_rate_scheduler(
        factors='constant * linear_warmup * rsqrt_decay',
        base_learning_rate=0.5,
        warmup_steps=1000,
        decay_factor=0.5,
        steps_per_decay=20000,
        steps_per_cycle=100000):
    """Creates learning rate schedule.

    Interprets factors in the factors string which can consist of:
    * constant: interpreted as the constant value,
    * linear_warmup: interpreted as linear warmup until warmup_steps,
    * rsqrt_decay: divide by square root of max(step, warmup_steps)
    * rsqrt_normalized_decay: divide by square root of max(step/warmup_steps, 1)
    * decay_every: Every k steps decay the learning rate by decay_factor.
    * cosine_decay: Cyclic cosine decay, uses steps_per_cycle parameter.

    Args:
        factors: string, factors separated by '*' that defines the schedule.
        base_learning_rate: float, the starting constant for the lr schedule.
        warmup_steps: int, how many steps to warm up for in the warmup schedule.
        decay_factor: float, the amount to decay the learning rate by.
        steps_per_decay: int, how often to decay the learning rate.
        steps_per_cycle: int, steps per cycle when using cosine decay.

    Returns:
        a function learning_rate(step): float -> {'learning_rate': float}, the
        step-dependent lr.
    """
    factors = [n.strip() for n in factors.split('*')]

    def step_fn(step):
        """Step to learning rate function."""
        ret = 1.0
        for name in factors:
            if name == 'constant':
                ret *= base_learning_rate
            elif name == 'linear_warmup':
                ret *= jnp.minimum(1.0, step / warmup_steps)
            elif name == 'rsqrt_decay':
                ret /= jnp.sqrt(jnp.maximum(step, warmup_steps))
            elif name == 'rsqrt_normalized_decay':
                ret *= jnp.sqrt(warmup_steps)
                ret /= jnp.sqrt(jnp.maximum(step, warmup_steps))
            elif name == 'decay_every':
                ret *= (decay_factor**(step // steps_per_decay))
            elif name == 'cosine_decay':
                progress = jnp.maximum(0.0,
                                                              (step - warmup_steps) / float(steps_per_cycle))
                ret *= jnp.maximum(0.0,
                                                      0.5 * (1.0 + jnp.cos(jnp.pi * (progress % 1.0))))
            else:
                raise ValueError('Unknown factor %s.' % name)
        return jnp.asarray(ret, dtype=jnp.float32)

    return step_fn


def create_optimiser(factors, base_learning_rate, warmup_steps, weight_decay):
    lr_fn = create_learning_rate_scheduler(factors=factors,
                                           base_learning_rate=base_learning_rate,
                                           warmup_steps=warmup_steps)

    @optax.inject_hyperparams
    def optim(learning_rate):
        return optax.adamw(learning_rate, b1=0.9, b2=0.98, eps=1e-9,
                           weight_decay=weight_decay)

    return optim(lr_fn)


def compute_weighted_cross_entropy(logits, targets, num_classes, weights=None):
    """Compute weighted cross entropy and entropy for log probs and targets.

    Args:
      logits: [batch, num_classes] float array.
      targets: categorical targets [batch, length] int array.
      num_classes: int, num classes of problem.
      weights: None or array of shape [batch x length]

    Returns:
        Tuple of scalar loss and batch normalizing factor.
    """
    onehot_targets = common_utils.onehot(targets, num_classes)
    loss = -jnp.sum(onehot_targets * nn.log_softmax(logits), axis=-1)
    normalizing_factor = onehot_targets.sum()
    if weights is not None:
        loss = loss * weights
        normalizing_factor = weights.sum()

    return loss.sum(), normalizing_factor


def compute_weighted_accuracy(logits, targets, weights=None):
    """Compute weighted accuracy for log probs and targets.

    Args:
      logits: [batch, num_classes] float array.
      targets: categorical targets [batch] int array.
      weights: None or array of shape [batch]

    Returns:
        Tuple of scalar accuracy and batch normalizing factor.
    """
    if logits.ndim != targets.ndim + 1:
        raise ValueError('Incorrect shapes. Got shape %s logits and %s targets' %
                                          (str(logits.shape), str(targets.shape)))
    loss = jnp.equal(jnp.argmax(logits, axis=-1), targets)
    normalizing_factor = onp.prod(logits.shape[:-1])
    if weights is not None:
        loss = loss * weights
        normalizing_factor = weights.sum()

    return loss.sum(), normalizing_factor


def compute_metrics(logits, labels, weights, *, num_classes):
    """Compute summary metrics."""
    loss, weight_sum = compute_weighted_cross_entropy(
            logits, labels, num_classes=num_classes, weights=None)
    acc, _ = compute_weighted_accuracy(logits, labels, weights)
    metrics = {
            'loss': loss,
            'accuracy': acc,
            'denominator': weight_sum,
    }
    metrics = jax.lax.psum(metrics, 'batch')
    return metrics


def stnd_get_loss_fn_and_targets(t_state, batch, dropout_rng, *, num_classes, model_kwargs=None):
    if model_kwargs is None:
        model_kwargs = {}

    keys = ['inputs', 'targets']
    (inputs, targets) = [batch.get(k, None) for k in keys]

    def loss_fn(params):
        """Loss function used for training."""
        logits = t_state.apply_fn({'params': params}, inputs, train=True,
                                  rngs={'dropout': dropout_rng}, **model_kwargs)
        loss, weight_sum = compute_weighted_cross_entropy(
                logits, targets, num_classes=num_classes, weights=None)
        mean_loss = loss / weight_sum
        return mean_loss, logits

    return loss_fn, targets


def train_step(t_state, batch, dropout_rng, *, num_classes, get_loss_fn_and_targets_fn=None,
               model_kwargs=None):
    """Perform a single training step."""
    # We handle PRNG splitting inside the top pmap, rather
    # than handling it outside in the training loop - doing the
    # latter can add some stalls to the devices.
    dropout_rng, new_dropout_rng = random.split(dropout_rng)

    if get_loss_fn_and_targets_fn is None:
        get_loss_fn_and_targets_fn = stnd_get_loss_fn_and_targets

    loss_fn, targets = get_loss_fn_and_targets_fn(t_state, batch, dropout_rng,
                                                  num_classes=num_classes,
                                                  model_kwargs=model_kwargs)

    (_, logits), grads = jax.value_and_grad(loss_fn, has_aux=True)(t_state.params)
    grads = jax.lax.pmean(grads, 'batch')
    new_t_state = t_state.apply_gradients(grads=grads)

    metrics = compute_metrics(logits, targets, None, num_classes=num_classes)
    metrics['learning_rate'] = t_state.opt_state.hyperparams['learning_rate']

    # Pack train state and return
    return new_t_state, metrics, new_dropout_rng


def stnd_get_logits_and_targets(t_state, batch, model_kwargs=None):
    if model_kwargs is None:
        model_kwargs = {}
    keys = ['inputs', 'targets']
    (inputs, targets) = [batch.get(k, None) for k in keys]
    logits = t_state.apply_fn({'params': t_state.params}, inputs, train=False, **model_kwargs)
    return logits, targets


def eval_step(t_state, batch, *, num_classes, get_logits_and_targets_fn=None, model_kwargs=None):
    if get_logits_and_targets_fn is None:
        get_logits_and_targets_fn = stnd_get_logits_and_targets
    logits, targets = get_logits_and_targets_fn(t_state, batch, model_kwargs)
    return compute_metrics(logits, targets, None, num_classes=num_classes)


def run_eval(eval_ds, t_state, p_eval_step, n_devices=None, num_eval_steps=-1):
    eval_metrics = []
    eval_iter = iter(eval_ds)
    if num_eval_steps == -1:
        num_iter = itertools.count()
    else:
        num_iter = range(num_eval_steps)
    for _, eval_batch in zip(num_iter, eval_iter):
        eval_batch = shard(tree_map(lambda x: x._numpy(), eval_batch), n_devices=n_devices)
        metrics = p_eval_step(t_state, eval_batch)
        eval_metrics.append(metrics)
    eval_metrics = common_utils.get_metrics(eval_metrics)
    eval_metrics_sums = tree_map(jnp.sum, eval_metrics)
    eval_denominator = eval_metrics_sums.pop('denominator')
    eval_summary = tree_map(lambda x: x / eval_denominator, eval_metrics_sums)
    # Calculate (clipped) perplexity after averaging log-perplexities:
    eval_summary['perplexity'] = jnp.clip(
            jnp.exp(eval_summary['loss']), a_max=1.0e4)
    return eval_summary


def main_train_eval(
        *, t_state, train_ds, eval_ds, test_ds, n_devices, gpu_devices, dropout_rngs, num_classes,
        num_train_steps, num_eval_steps, model_dir, save_checkpoints, restore_checkpoints,
        checkpoint_freq, eval_freq, test_only, test_on_eval=False, get_loss_fn_and_targets_fn=None,
        get_logits_and_targets_fn=None):

    start_step = 0
    if restore_checkpoints or test_only:
        # Restore unreplicated optimizer + model state from last checkpoint.
        t_state = checkpoints.restore_checkpoint(model_dir, t_state)
        # Grab last step.
        start_step = t_state.step

    # Replicate t_state onto gpu
    t_state = jax_utils.replicate(t_state, devices=gpu_devices)

    p_train_step = jax.pmap(partial(train_step, num_classes=num_classes,
                                    get_loss_fn_and_targets_fn=get_loss_fn_and_targets_fn),
                            axis_name='batch', devices=gpu_devices)
    p_eval_step = jax.pmap(partial(eval_step, num_classes=num_classes,
                                   get_logits_and_targets_fn=get_logits_and_targets_fn),
                           axis_name='batch', devices=gpu_devices)
    # p_pred_step = jax.pmap(predict_step, axis_name='batch', devices=gpu_devices)

    if test_only:
        with tf.io.gfile.GFile(os.path.join(model_dir, 'results.json'), 'w') as f:
            test_summary = run_eval(test_ds, t_state, p_eval_step, n_devices=n_devices)
            json.dump(tree_map(lambda x: x.tolist(), test_summary), f)
        return

    if jax.process_index() == 0:
        summary_writer = tensorboard.SummaryWriter(
                os.path.join(model_dir, 'summary'))

    metrics_all = []
    tick = time.time()
    train_iter = iter(train_ds)
    logging.info('Starting training')
    logging.info('====================')

    for step, batch in zip(range(start_step, num_train_steps), train_iter):
        batch = shard(tree_map(lambda x: x._numpy(), batch), n_devices=n_devices)
        t_state, metrics, dropout_rngs = p_train_step(t_state, batch, dropout_rngs)
        metrics_all.append(metrics)
        logging.info('train in step: %d', step)

        # Save a Checkpoint
        if ((step % checkpoint_freq == 0 and step > 0) or
                step == num_train_steps - 1):
            if jax.process_index() == 0 and save_checkpoints:
                # Save unreplicated optimizer + model state.
                checkpoints.save_checkpoint(model_dir, jax_utils.unreplicate(t_state), step)

        # Periodic metric handling.
        if step % eval_freq == 0 and step > 0:
            metrics_all = common_utils.get_metrics(metrics_all)
            lr = metrics_all.pop('learning_rate').mean()
            metrics_sums = tree_map(jnp.sum, metrics_all)
            denominator = metrics_sums.pop('denominator')
            summary = tree_map(lambda x: x / denominator, metrics_sums)  # pylint: disable=cell-var-from-loop
            summary['learning_rate'] = lr
            # Calculate (clipped) perplexity after averaging log-perplexities:
            summary['perplexity'] = jnp.clip(jnp.exp(summary['loss']), a_max=1.0e4)
            logging.info(f"train in step: {step}, loss: {r4(summary['loss'])}, acc: {r4(summary['accuracy'])}")
            if jax.process_index() == 0:
                tock = time.time()
                steps_per_sec = eval_freq / (tock - tick)
                tick = tock
                summary_writer.scalar('steps per second', steps_per_sec, step)
                for key, val in summary.items():
                    summary_writer.scalar(f'train_{key}', val, step)
                summary_writer.flush()
            # Reset metric accumulation for next evaluation cycle.
            metrics_all = []

            # Eval Metrics
            eval_summary = run_eval(eval_ds, t_state, p_eval_step, n_devices=n_devices,
                                                num_eval_steps=num_eval_steps)
            logging.info('eval in step: %d, loss: %.4f, acc: %.4f', step,
                                      eval_summary['loss'], eval_summary['accuracy'])
            if jax.process_index() == 0:
                for key, val in eval_summary.items():
                    summary_writer.scalar(f'eval_{key}', val, step)
                summary_writer.flush()

            if test_on_eval:
                # Test eval
                # Eval Metrics
                logging.info('Testing...')
                test_summary = run_eval(test_ds, t_state, p_eval_step, n_devices=n_devices,
                                        num_eval_steps=num_eval_steps)
                logging.info('test in step: %d, loss: %.4f, acc: %.4f', step,
                                          test_summary['loss'], test_summary['accuracy'])
                if jax.process_index() == 0:
                    for key, val in test_summary.items():
                        summary_writer.scalar(f'test_{key}', val, step)
                    summary_writer.flush()

