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
"""Document Classification tasks."""
from functools import partial
import itertools
import json
import os
import time

from absl import app
from absl import flags
from absl import logging
from flax import jax_utils
from flax.metrics import tensorboard
from flax.training import checkpoints, common_utils, train_state
import optax
import jax
from jax import random
from jax.tree_util import tree_map
import jax.numpy as jnp

from lra_benchmarks.text_classification import input_pipeline
from lra_benchmarks.utils import train_utils
from lra_benchmarks.utils.device_utils import get_devices, shard

from ml_collections import config_flags
import tensorflow.compat.v2 as tf


FLAGS = flags.FLAGS

config_flags.DEFINE_config_file(
        'config', None, 'Training configuration.', lock_config=True)
flags.DEFINE_string(
        'model_dir', default=None, help='Directory to store model data.')
flags.DEFINE_string(
        'data_dir', default=None, help='Directory containing datasets.')
flags.DEFINE_bool(
        'test_only', default=False, help='Run the evaluation on the test data.')

CLASS_MAP = {'imdb_reviews': 2, 'yelp_reviews': 2, 'agnews': 2}


def create_train_state(flax_module, model_kwargs, init_rng, input_shape, tx
                       ) -> train_state.TrainState:
    """Creates and initializes the model."""

    @partial(jax.jit, backend='cpu')
    def _create_train_state(init_rng):
        module = flax_module(**model_kwargs)
        variables = module.init(init_rng, jnp.ones(input_shape), train=False)
        params = variables['params']
        return train_state.TrainState.create(apply_fn=module.apply, params=params, tx=tx)

    return _create_train_state(init_rng)


def compute_metrics(logits, labels, weights, *, num_classes):
    """Compute summary metrics."""
    loss, weight_sum = train_utils.compute_weighted_cross_entropy(
            logits, labels, num_classes=num_classes, weights=None)
    acc, _ = train_utils.compute_weighted_accuracy(logits, labels, weights)
    metrics = {
            'loss': loss,
            'accuracy': acc,
            'denominator': weight_sum,
    }
    metrics = jax.lax.psum(metrics, 'batch')
    return metrics


def train_step(t_state, batch, *, num_classes, dropout_rng=None):
    """Perform a single training step."""
    train_keys = ['inputs', 'targets']
    (inputs, targets) = [batch.get(k, None) for k in train_keys]

    # We handle PRNG splitting inside the top pmap, rather
    # than handling it outside in the training loop - doing the
    # latter can add some stalls to the devices.
    dropout_rng, new_dropout_rng = random.split(dropout_rng)

    def loss_fn(params):
        """Loss function used for training."""
        logits = t_state.apply_fn({'params': params}, inputs, train=True,
                                  rngs={'dropout': dropout_rng})
        loss, weight_sum = train_utils.compute_weighted_cross_entropy(
                logits, targets, num_classes=num_classes, weights=None)
        mean_loss = loss / weight_sum
        return mean_loss, logits

    (_, logits), grads = jax.value_and_grad(loss_fn, has_aux=True)(t_state.params)
    grads = jax.lax.pmean(grads, 'batch')
    new_t_state = t_state.apply_gradients(grads=grads)

    metrics = compute_metrics(logits, targets, None, num_classes=num_classes)
    metrics['learning_rate'] = t_state.opt_state.hyperparams['learning_rate']

    # Pack train state and return
    return new_t_state, metrics, new_dropout_rng


def eval_step(t_state, batch, *, num_classes):
    eval_keys = ['inputs', 'targets']
    (inputs, targets) = [batch.get(k, None) for k in eval_keys]
    logits = t_state.apply_fn({'params': t_state.params}, inputs, train=False)
    logging.info(logits)
    return compute_metrics(logits, targets, None, num_classes=num_classes)


def main(argv):
    if len(argv) > 1:
        raise app.UsageError('Too many command-line arguments.')

    tf.enable_v2_behavior()

    config = FLAGS.config
    logging.info('===========Config Dict============')
    logging.info(config)
    batch_size = config.batch_size
    learning_rate = config.learning_rate
    num_train_steps = config.num_train_steps
    num_eval_steps = config.num_eval_steps
    eval_freq = config.eval_frequency
    random_seed = config.random_seed
    model_type = config.model_type
    num_classes = CLASS_MAP[config.task_name]
    gpu_devices, n_devices = get_devices(config.available_devices)

    logging.info(f"GPU devices: {gpu_devices}")

    max_length = config.max_length

    if jax.process_index() == 0:
        summary_writer = tensorboard.SummaryWriter(
                os.path.join(FLAGS.model_dir, 'summary'))

    if batch_size % n_devices > 0:
        raise ValueError('Batch size must be divisible by the number of devices')

    train_ds, eval_ds, test_ds, encoder = input_pipeline.get_tc_datasets(
            n_devices=n_devices,
            task_name=config.task_name,
            data_dir=FLAGS.data_dir,
            batch_size=batch_size,
            fixed_vocab=None,
            max_length=max_length,
            num_data_entries=config.num_data_entries)

    vocab_size = encoder.vocab_size
    logging.info('Vocab Size: %d', vocab_size)

    train_ds = train_ds.repeat()

    train_iter = iter(train_ds)
    input_shape = (batch_size, max_length)

    model_kwargs = {
        'vocab_size': vocab_size,
        'emb_dim': config.emb_dim,
        'num_heads': config.num_heads,
        'num_layers': config.num_layers,
        'qkv_dim': config.qkv_dim,
        'mlp_dim': config.mlp_dim,
        'max_len': max_length,
        'classifier': True,
        'num_classes': num_classes,
        'classifier_pool': config.classifier_pool
    }

    rng = random.PRNGKey(random_seed)
    rng = jax.random.fold_in(rng, jax.process_index())
    rng, init_rng = random.split(rng)
    # We init the first set of train PRNG keys, but update it afterwards inside
    # the main pmap'd training update for performance.
    dropout_rngs = random.split(rng, n_devices)

    lr_fn = train_utils.create_learning_rate_scheduler(factors=config.factors,
                                                       base_learning_rate=learning_rate,
                                                       warmup_steps=config.warmup)

    @optax.inject_hyperparams
    def optim(learning_rate):
        return optax.adamw(learning_rate, b1=0.9, b2=0.98, eps=1e-9,
                           weight_decay=FLAGS.config.weight_decay)

    tx = optim(lr_fn)

    t_state = train_utils.get_model(model_type, create_train_state, model_kwargs, init_rng,
                                    input_shape, tx)

    start_step = 0
    if config.restore_checkpoints or FLAGS.test_only:
        # Restore unreplicated optimizer + model state from last checkpoint.
        t_state = checkpoints.restore_checkpoint(FLAGS.model_dir, t_state)
        # Grab last step.
        start_step = t_state.step

    # Replicate t_state, not sure if needed
    t_state = jax_utils.replicate(t_state, devices=gpu_devices)

    p_train_step = jax.pmap(partial(train_step, num_classes=num_classes),
                            axis_name='batch', devices=gpu_devices)
    p_eval_step = jax.pmap(partial(eval_step, num_classes=num_classes),
                           axis_name='batch', devices=gpu_devices)
    # p_pred_step = jax.pmap(predict_step, axis_name='batch', devices=gpu_devices)

    def run_eval(eval_ds, num_eval_steps=-1):
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
        eval_summary = tree_map(
                lambda x: x / eval_denominator,  # pylint: disable=cell-var-from-loop
                eval_metrics_sums)
        # Calculate (clipped) perplexity after averaging log-perplexities:
        eval_summary['perplexity'] = jnp.clip(
                jnp.exp(eval_summary['loss']), a_max=1.0e4)
        return eval_summary

    if FLAGS.test_only:
        with tf.io.gfile.GFile(os.path.join(FLAGS.model_dir, 'results.json'),
                                                      'w') as f:
            test_summary = run_eval(test_ds)
            json.dump(tree_map(lambda x: x.tolist(), test_summary), f)
        return

    metrics_all = []
    tick = time.time()
    logging.info('Starting training')
    logging.info('====================')

    for step, batch in zip(range(start_step, num_train_steps), train_iter):
        batch = shard(tree_map(lambda x: x._numpy(), batch), n_devices=n_devices)
        t_state, metrics, dropout_rngs = p_train_step(t_state, batch, dropout_rng=dropout_rngs)
        metrics_all.append(metrics)
        logging.info('train in step: %d', step)

        # Save a Checkpoint
        if ((step % config.checkpoint_freq == 0 and step > 0) or
                step == num_train_steps - 1):
            if jax.process_index() == 0 and config.save_checkpoints:
                # Save unreplicated optimizer + model state.
                checkpoints.save_checkpoint(FLAGS.model_dir, jax_utils.unreplicate(t_state),
                                            step)

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
            logging.info('train in step: %d, loss: %.4f, acc: %.4f', step,
                                      summary['loss'], summary['accuracy'])
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
            eval_summary = run_eval(eval_ds, num_eval_steps)
            logging.info('eval in step: %d, loss: %.4f, acc: %.4f', step,
                                      eval_summary['loss'], eval_summary['accuracy'])
            if jax.process_index() == 0:
                for key, val in eval_summary.items():
                    summary_writer.scalar(f'eval_{key}', val, step)
                summary_writer.flush()


if __name__ == '__main__':
    app.run(main)
