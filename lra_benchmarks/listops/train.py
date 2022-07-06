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
"""Main training script for the listops task."""
from functools import partial
import itertools
import json
import os
import time

from absl import app
from absl import flags
from absl import logging

from flax import jax_utils
import flax.linen as nn
from flax.metrics import tensorboard
from flax.training import checkpoints
from flax.training import common_utils

import jax
from jax import random
from jax.tree_util import tree_map
import jax.nn as jnn
import jax.numpy as jnp

import optax
from ml_collections import config_flags
import numpy as np
import tensorflow.compat.v2 as tf

from lra_benchmarks.listops import input_pipeline
from lra_benchmarks.utils import train_utils
from lra_benchmarks.utils.device_utils import get_devices, shard


FLAGS = flags.FLAGS

config_flags.DEFINE_config_file('config', None, 'Training configuration.', lock_config=True)
flags.DEFINE_string('model_dir', default=None, help='Directory to store model data.')
flags.DEFINE_string('data_dir', default=None, help='Directory containing datasets.')
flags.DEFINE_bool('test_only', default=False, help='Run the evaluation on the test data.')


def tohost(x):
    """Collect batches from all devices to host and flatten batch dimensions."""
    n_device, n_batch, *remaining_dims = x.shape
    return np.array(x).reshape((n_device * n_batch,) + tuple(remaining_dims))


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
    max_length = config.max_length
    gpu_devices, n_devices = get_devices(config.available_devices)
    model_kwargs = (config.model_kwargs.to_dict() if 'model_kwargs' in config else {})

    logging.info(f"GPU devices: {gpu_devices}")

    if jax.process_index() == 0:
        summary_writer = tensorboard.SummaryWriter(
                os.path.join(FLAGS.model_dir, 'summary'))

    if batch_size % jax.device_count() > 0:
        raise ValueError('Batch size must be divisible by the number of devices')

    train_ds, eval_ds, test_ds, encoder = input_pipeline.get_datasets(
            n_devices=n_devices,
            task_name=config.task_name,
            data_dir=FLAGS.data_dir,
            batch_size=batch_size,
            max_length=max_length)

    train_ds = train_ds.repeat()
    train_iter = iter(train_ds)
    input_shape = (batch_size, max_length)
    vocab_size = encoder.vocab_size
    logging.info('Vocab Size: %d', vocab_size)

    model_kwargs.update({
            'vocab_size': vocab_size,
            'emb_dim': config.emb_dim,
            'num_heads': config.num_heads,
            'num_layers': config.num_layers,
            'qkv_dim': config.qkv_dim,
            'mlp_dim': config.mlp_dim,
            'max_len': max_length,
            'classifier': True,
            'num_classes': 10
    })

    rng = random.PRNGKey(random_seed)
    rng = jax.random.fold_in(rng, jax.process_index())
    rng, init_rng = random.split(rng)
    # We init the first set of dropout PRNG keys, but update it afterwards inside
    # the main pmap'd training update for performance.
    dropout_rngs = random.split(rng, n_devices)

    tx = train_utils.create_optimiser(config.factors, learning_rate, config.warmup,
                                      FLAGS.config.weight_decay)

    t_state = train_utils.get_model(model_type, train_utils.create_train_state, model_kwargs,
                                    init_rng, input_shape, tx)

    start_step = 0
    if config.restore_checkpoints or FLAGS.test_only:
        # Restore unreplicated optimizer + model state from last checkpoint.
        t_state = checkpoints.restore_checkpoint(FLAGS.model_dir, t_state)
        # Grab last step.
        start_step = t_state.step

    # Replicate t_state, not sure if needed
    t_state = jax_utils.replicate(t_state, devices=gpu_devices)

    p_train_step = jax.pmap(partial(train_utils.train_step, num_classes=10),
                            axis_name='batch', devices=gpu_devices)
    p_eval_step = jax.pmap(partial(train_utils.eval_step, num_classes=10),
                           axis_name='batch', devices=gpu_devices)
    # p_pred_step = jax.pmap(predict_step, axis_name='batch', devices=gpu_devices)

    if FLAGS.test_only:
        with tf.io.gfile.GFile(os.path.join(FLAGS.model_dir, 'results.json'),
                                                      'w') as f:
            test_summary = train_utils.run_eval(test_ds, t_state, p_eval_step, n_devices=n_devices)
            json.dump(tree_map(lambda x: x.tolist(), test_summary), f)
        return

    metrics_all = []
    tick = time.time()
    logging.info('Starting training')
    logging.info('====================')

    for step, batch in zip(range(start_step, num_train_steps), train_iter):
        batch = common_utils.shard(tree_map(lambda x: x._numpy(), batch))  # pylint: disable=protected-access
        optimizer, metrics, dropout_rngs = p_train_step(
                optimizer, batch, dropout_rng=dropout_rngs)
        metrics_all.append(metrics)
        logging.info('train in step: %d', step)

        # Save a Checkpoint
        if ((step % config.checkpoint_freq == 0 and step > 0) or
                step == num_train_steps - 1):
            if jax.process_index() == 0 and config.save_checkpoints:
                # Save unreplicated optimizer + model state.
                checkpoints.save_checkpoint(FLAGS.model_dir, jax_utils.unreplicate(optimizer), step)

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
            logging.info('train in step: %d, loss: %.4f', step, summary['loss'])
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
            eval_summary = train_utils.run_eval(eval_ds, t_state, p_eval_step, n_devices=n_devices,
                                                num_eval_steps=num_eval_steps)
            logging.info('eval in step: %d, loss: %.4f, acc: %.4f', step,
                                      eval_summary['loss'], eval_summary['accuracy'])
            if jax.process_index() == 0:
                for key, val in eval_summary.items():
                    summary_writer.scalar(f'eval_{key}', val, step)
                summary_writer.flush()


if __name__ == '__main__':
    app.run(main)
