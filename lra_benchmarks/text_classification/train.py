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
from absl import app, logging, flags
from pprint import pformat

import jax
from jax import random

import tensorflow.compat.v2 as tf

from lra_benchmarks.text_classification import input_pipeline
from lra_benchmarks.utils import train_utils
from lra_benchmarks.utils.device_utils import get_devices
from lra_benchmarks.utils.config_utils import load_configs


FLAGS = flags.FLAGS

flags.DEFINE_string('model_dir', default=None, help='Directory to store model data.')
flags.DEFINE_bool('test_only', default=False, help='Run the evaluation on the test data.')
flags.DEFINE_list('config_paths', default=None, help="Config files, can specify many and they will overwrite with last given having highest priority")


CLASS_MAP = {'imdb_reviews': 2, 'yelp_reviews': 2, 'agnews': 2}


def main(argv):
    if len(argv) > 1:
        raise app.UsageError('Too many command-line arguments.')

    tf.get_logger().setLevel('ERROR')
    tf.enable_v2_behavior()

    logging.info("===========Config Paths===========\n" + pformat(FLAGS.config_paths))
    config = load_configs(FLAGS.config_paths)
    logging.info("===========Config Dict============\n" + pformat(config))

    batch_size = config["batch_size"]
    learning_rate = config["learning_rate"]
    num_train_steps = config["num_train_steps"]
    num_eval_steps = config["num_eval_steps"]
    eval_freq = config["eval_frequency"]
    random_seed = config["random_seed"]
    base_type = config["base_type"]
    _model_type = config["model_type"]
    num_classes = CLASS_MAP[config["task_name"]]
    max_length = config["max_length"]

    gpu_devices, n_devices = get_devices(config["available_devices"])
    logging.info(f"GPU devices: {gpu_devices}")

    if batch_size % n_devices > 0:
        raise ValueError('Batch size must be divisible by the number of devices')

    if base_type == "encoder":
        model_type = _model_type
    elif base_type == "dual_encoder":
        model_type = _model_type + "_dual"
    else:
        raise ValueError("Bad base_type, should be encoder or dual_encoder")

    train_ds, eval_ds, test_ds, encoder = input_pipeline.get_tc_datasets(
            n_devices=n_devices,
            task_name=config["task_name"],
            data_dir=config["data_dir"],
            batch_size=batch_size,
            fixed_vocab=None,
            max_length=max_length,
            num_data_entries=config["num_data_entries"])

    train_ds = train_ds.repeat()
    input_shape = (batch_size, max_length)
    vocab_size = encoder.vocab_size
    logging.info('Vocab Size: %d', vocab_size)

    model_kwargs = {
        'vocab_size': vocab_size,
        'emb_dim': config["emb_dim"],
        'num_heads': config["num_heads"],
        'num_layers': config["num_layers"],
        'qkv_dim': config["qkv_dim"],
        'mlp_dim': config["mlp_dim"],
        'max_len': max_length,
        'classifier': True,
        'num_classes': num_classes,
        'classifier_pool': config["classifier_pool"]
    }

    rng = random.PRNGKey(random_seed)
    rng = jax.random.fold_in(rng, jax.process_index())
    rng, init_rng = random.split(rng)
    # We init the first set of train PRNG keys, but update it afterwards inside
    # the main pmap'd training update for performance.
    dropout_rngs = random.split(rng, n_devices)

    tx = train_utils.create_optimiser(config["factors"], learning_rate, config["warmup"],
                                      config["weight_decay"])

    t_state = train_utils.get_model(model_type, train_utils.create_train_state, model_kwargs,
                                    init_rng, [input_shape], tx)

    train_utils.main_train_eval(
        t_state=t_state,
        train_ds=train_ds,
        eval_ds=eval_ds,
        test_ds=test_ds,
        n_devices=n_devices,
        gpu_devices=gpu_devices,
        dropout_rngs=dropout_rngs,
        num_classes=num_classes,
        num_train_steps=num_train_steps,
        num_eval_steps=num_eval_steps,
        model_dir=FLAGS.model_dir,
        save_checkpoints=config["save_checkpoints"],
        restore_checkpoints=config["restore_checkpoints"],
        checkpoint_freq=config["checkpoint_freq"],
        eval_freq=eval_freq,
        test_only=FLAGS.test_only,
    )


if __name__ == "__main__":
    app.run(main)

