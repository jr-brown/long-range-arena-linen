from absl import app, logging, flags
from pprint import pformat
from functools import partial

import json
import os
from datetime import datetime

import jax
from jax import random
from jax.tree_util import tree_map

from flax import jax_utils
from flax.training import checkpoints

import tensorflow.compat.v2 as tf

from lra_benchmarks.text_classification import input_pipeline as tc_input_pipeline
from lra_benchmarks.listops import input_pipeline as listops_input_pipeline
from lra_benchmarks.matching import input_pipeline as matching_input_pipeline

from lra_benchmarks.utils import train_utils
from lra_benchmarks.utils.device_utils import get_devices
from lra_benchmarks.utils.config_utils import load_configs


def get_time_stamp():
    date = datetime.now().date().__str__()
    time = datetime.now().time().__str__()
    time = '-'.join(time.split(':')[:2])
    return f"{date}_{time}"


flags.DEFINE_list('config_paths', default=None, help="Config files, can specify many and they will overwrite with last given having highest priority")


def matching_task_get_loss_fn_and_targets(t_state, batch, dropout_rng, *, num_classes,
                                          model_kwargs=None):
    if model_kwargs is None:
        model_kwargs = {}

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


def matching_task_get_logits_and_targets(t_state, batch, model_kwargs=None):
    if model_kwargs is None:
        model_kwargs = {}

    keys = ['inputs1', 'inputs2', 'targets']
    (inputs1, inputs2, targets) = [batch.get(k, None) for k in keys]
    logits = t_state.apply_fn({'params': t_state.params}, inputs1, inputs2, train=False)
    return logits, targets


def main(argv):
    if len(argv) > 1:
        raise app.UsageError('Too many command-line arguments.')

    tf.get_logger().setLevel('ERROR')
    tf.enable_v2_behavior()

    logging.info("========== Config Paths ==========\n" + pformat(flags.FLAGS.config_paths))

    config = load_configs(flags.FLAGS.config_paths)
    logging.info("========== Config Dict ===========\n" + pformat(config))

    model_kwargs = config["model_kwargs"]
    num_classes = model_kwargs["num_classes"]
    max_len = model_kwargs["max_len"]

    data_kwargs = config["data_kwargs"]
    batch_size = data_kwargs["batch_size"]

    optim_kwargs = config["optim_kwargs"]

    num_train_steps = config["num_train_steps"]
    num_eval_steps = config["num_eval_steps"]
    eval_freq = config["eval_frequency"]
    random_seed = config["random_seed"]
    model_base = config["model_base"]
    model_type = config["model_type"]
    task_type = config["task_type"]
    save_checkpoints = config["save_checkpoints"]
    restore_checkpoints = config["restore_checkpoints"]
    checkpoint_freq = config["checkpoint_freq"]
    available_devices = config["available_devices"]
    model_folder = config["model_folder"]
    test_only = config["test_only"]
    test_on_eval = config["test_on_eval"]
    output_db_path = config.get("output_db_path", None)

    model_dir = os.path.join(model_folder, model_type)
    model_key = model_type + "_" + model_base
    model_db_name = f"{task_type}_{model_type}_{get_time_stamp()}"

    gpu_devices, n_devices = get_devices(available_devices)
    logging.info(f"GPU devices: {gpu_devices}")

    input_shape = (batch_size, max_len)

    if batch_size % n_devices > 0:
        raise ValueError('Batch size must be divisible by the number of devices')

    # Use defaults in train_utils unless doing matching task
    loss_targ_fn = None
    logi_targ_fn = None

    # Default
    input_shapes = [input_shape]

    if task_type == "text_classification":
        train_ds, eval_ds, test_ds, encoder = tc_input_pipeline.get_tc_datasets(
            n_devices=n_devices,
            fixed_vocab=None,
            max_length=max_len,
            **data_kwargs)
    elif task_type == "listops":
        train_ds, eval_ds, test_ds, encoder = listops_input_pipeline.get_datasets(
            n_devices=n_devices,
            max_length=max_len,
            **data_kwargs)
    elif task_type == "matching":
        train_ds, eval_ds, test_ds, encoder = matching_input_pipeline.get_matching_datasets(
            n_devices=n_devices,
            fixed_vocab=None,
            max_length=max_len,
            **data_kwargs)
        loss_targ_fn = matching_task_get_loss_fn_and_targets
        logi_targ_fn = matching_task_get_logits_and_targets
        input_shapes = [input_shape, input_shape]
    else:
        raise ValueError(f"Task type {task_type} is unsupported")

    train_ds = train_ds.repeat()
    vocab_size = encoder.vocab_size
    logging.info(f"Vocab Size: {vocab_size}")

    model_kwargs.update({
        'vocab_size': vocab_size,
        'classifier': True,
    })
    logging.info("======= Final Model Kwargs =======\n" + pformat(model_kwargs))

    rng = random.PRNGKey(random_seed)
    rng = jax.random.fold_in(rng, jax.process_index())
    rng, init_rng = random.split(rng)
    # We init the first set of train PRNG keys, but update it afterwards inside
    # the main pmap'd training update for performance.
    dropout_rngs = random.split(rng, n_devices)

    tx = train_utils.create_optimiser(**optim_kwargs)
    t_state = train_utils.get_model(model_key, model_kwargs, init_rng, input_shapes, tx)

    start_step = 0
    if restore_checkpoints or test_only:
        # Restore unreplicated optimizer + model state from last checkpoint.
        t_state = checkpoints.restore_checkpoint(model_dir, t_state)
        # Grab last step.
        start_step = t_state.step

    # Replicate t_state onto gpu
    t_state = jax_utils.replicate(t_state, devices=gpu_devices)

    p_train_step = jax.pmap(partial(train_utils.train_step, num_classes=num_classes,
                                    get_loss_fn_and_targets_fn=loss_targ_fn),
                            axis_name='batch', devices=gpu_devices)
    p_eval_step = jax.pmap(partial(train_utils.eval_step, num_classes=num_classes,
                                   get_logits_and_targets_fn=logi_targ_fn),
                           axis_name='batch', devices=gpu_devices)

    if test_only:
        with tf.io.gfile.GFile(os.path.join(model_dir, 'results.json'), 'w') as f:
            test_summary = train_utils.run_eval(test_ds, t_state, p_eval_step, n_devices=n_devices)
            json.dump(tree_map(lambda x: x.tolist(), test_summary), f)
        return

    # Initialise history dict
    history = {
        "train": {k : [] for k in ['learning_rate', 'perplexity', 'loss', 'accuracy',
                                   'steps_per_second']},
        "validation": {k: [] for k in ['perplexity', 'loss', 'accuracy']}
    }

    metrics_all = []
    logging.info('======= Starting training ========')

    t_state, _, metrics_all, history = train_utils.train(
        start_step=start_step,
        num_train_steps=num_train_steps,
        num_eval_steps=num_eval_steps,
        train_ds=train_ds,
        eval_ds=eval_ds,
        n_devices=n_devices,
        p_train_step=p_train_step,
        p_eval_step=p_eval_step,
        t_state=t_state,
        dropout_rngs=dropout_rngs,
        metrics_all=metrics_all,
        history=history,
        checkpoint_freq=checkpoint_freq,
        save_checkpoints=save_checkpoints,
        model_dir=model_dir,
        eval_freq=eval_freq
    )

    if output_db_path is not None:
        logging.info("Saving metrics and config data...")

        try:
            with open(output_db_path) as f:
                output_db = json.load(f)
        except FileNotFoundError:
            output_db = {}

        if model_db_name in output_db.keys():
            logging.warning("")

        output_db[model_db_name] = {
            "model_dir": model_dir,
            "config": config,
            "history": history
        }

        # Try to save the history
        json_exception = None
        with open(output_db_path, 'w', encoding="utf-8") as f:
            try:
                json.dump(output_db, f, ensure_ascii=False, indent=4)

            # Error writing a value to the json file
            except TypeError as e:
                json_exception = e

        # We want to clear the problematic data, record there was an error (by assigning null)
        # and then resave the output_db so it can cleanly save and still be valid json
        # After resaving the exception is raised as we are at end of program and want
        # diagnostics in the logs / output
        if json_exception is not None:
            output_db[model_db_name] = None
            with open(output_db_path, 'w', encoding="utf-8") as f:
                json.dump(output_db, f, ensure_ascii=False, indent=4)
            raise json_exception


if __name__ == "__main__":
    app.run(main)

