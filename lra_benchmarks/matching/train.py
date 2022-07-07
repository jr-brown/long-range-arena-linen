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
"""Main script for document matching in dual encoder style with AAN dataset."""
from absl import app
from absl import flags
from absl import logging

import jax
from jax import random

from ml_collections import config_flags
import tensorflow.compat.v2 as tf

from lra_benchmarks.matching import input_pipeline
from lra_benchmarks.utils import train_utils
from lra_benchmarks.utils.device_utils import get_devices


FLAGS = flags.FLAGS

config_flags.DEFINE_config_file('config', None, 'Training configuration.', lock_config=True)
flags.DEFINE_string('model_dir', default=None, help='Directory to store model data.')
flags.DEFINE_bool('test_only', default=False, help='Run the evaluation on the test data.')


def get_loss_fn_and_targets(t_state, batch, dropout_rng, *, num_classes):
    train_keys = ['inputs1', 'inputs2', 'targets']
    (inputs1, inputs2, targets) = [batch.get(k, None) for k in train_keys]

    def loss_fn(params):
        """Loss function used for training."""
        logits = t_state.apply_fn({'params': params}, inputs1, inputs2, train=True,
                                  rngs={'dropout': dropout_rng})
        loss, weight_sum = train_utils.compute_weighted_cross_entropy(
                logits, targets, num_classes=num_classes, weights=None)
        mean_loss = loss / weight_sum
        return mean_loss, logits

    return loss_fn, targets


def get_logits_and_targets(t_state, batch):
    keys = ['inputs1', 'inputs2', 'targets']
    (inputs1, inputs2, targets) = [batch.get(k, None) for k in keys]
    logits = t_state.apply_fn({'params': t_state.params}, inputs1, inputs2, train=False)
    return logits, targets


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

    logging.info(f"GPU devices: {gpu_devices}")

    if batch_size % jax.device_count() > 0:
        raise ValueError('Batch size must be divisible by the number of devices')

    train_ds, eval_ds, test_ds, encoder = input_pipeline.get_matching_datasets(
            n_devices=jax.local_device_count(),
            task_name=config.task_name,
            data_dir=config.data_dir,
            batch_size=batch_size,
            fixed_vocab=None,
            max_length=max_length,
            tokenizer=config.tokenizer,
            vocab_file_path=config.vocab_file_path)


    train_ds = train_ds.repeat()
    input_shape = (batch_size, max_length)
    vocab_size = encoder.vocab_size
    logging.info('Vocab Size: %d', vocab_size)

    model_kwargs = {
            'vocab_size': vocab_size,
            'emb_dim': config.emb_dim,
            'num_heads': config.num_heads,
            'num_layers': config.num_layers,
            'qkv_dim': config.qkv_dim,
            'mlp_dim': config.mlp_dim,
            'max_len': max_length,
            'classifier': True,
            'num_classes': 2,
            'classifier_pool': config.pooling_mode
    }

    rng = random.PRNGKey(random_seed)
    rng = jax.random.fold_in(rng, jax.process_index())
    rng, init_rng = random.split(rng)
    # We init the first set of train PRNG keys, but update it afterwards inside
    # the main pmap'd training update for performance.
    dropout_rngs = random.split(rng, n_devices)

    tx = train_utils.create_optimiser(config.factors, learning_rate, config.warmup,
                                      FLAGS.config.weight_decay)

    t_state = train_utils.get_model(model_type, train_utils.create_train_state, model_kwargs,
                                    init_rng, input_shape, tx)

    train_utils.main_train_eval(
        t_state=t_state,
        train_ds=train_ds,
        eval_ds=eval_ds,
        test_ds=test_ds,
        n_devices=n_devices,
        gpu_devices=gpu_devices,
        dropout_rngs=dropout_rngs,
        num_classes=2,
        num_train_steps=num_train_steps,
        num_eval_steps=num_eval_steps,
        model_dir=FLAGS.model_dir,
        save_checkpoints=config.save_checkpoints,
        restore_checkpoints=config.restore_checkpoints,
        checkpoint_freq=config.checkpoint_freq,
        eval_freq=eval_freq,
        test_only=FLAGS.test_only,
        test_on_eval=True,
        get_loss_fn_and_targets_fn=get_loss_fn_and_targets,
        get_logits_and_targets_fn=get_logits_and_targets,
    )


if __name__ == '__main__':
    app.run(main)
