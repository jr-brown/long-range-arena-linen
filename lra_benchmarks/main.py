# Long Range Arena Linen
# Copyright (C) 2022  Jason Brown
# 
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
# 
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
# 
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

from absl import app, logging, flags
from pprint import pformat
from functools import partial

import os
from datetime import datetime

import jax
from jax import random

from flax import jax_utils
from flax.training import checkpoints

from lra_benchmarks.utils import train_utils
from lra_benchmarks.utils.device_utils import get_devices
from lra_benchmarks.utils.config_utils import load_many_configs_into_one
from lra_benchmarks.utils.misc_utils import write_to_output_db
from lra_benchmarks.utils.pipeline_utils import get_datasets_and_encoder, get_task_fns_and_shape


def get_time_stamp():
    date = datetime.now().date().__str__()
    time = datetime.now().time().__str__()
    time = '-'.join(time.split(':')[:2])
    return f"{date}_{time}"


flags.DEFINE_list("config_paths", default=None, help="Config files, can specify many and they will overwrite with last given having highest priority")
flags.DEFINE_bool("dry_run", default=False, help="Just determine configs and setup but exit before running the model")


def main(argv):
    if len(argv) > 1:
        raise app.UsageError("Too many command-line arguments.")

    if flags.FLAGS.dry_run:
        logging.info("Executing dry run...")

    # Need to get this first to know log file name
    config = load_many_configs_into_one(flags.FLAGS.config_paths)
    model_type = config["model_type"]
    task_type = config["task_type"]
    run_name_suffix = config.get("run_name_suffix")
    run_name = config.get("existing_run_name")

    if run_name is None:
        if run_name_suffix is not None:
            run_name = f"{task_type}_{model_type}_{run_name_suffix}_{get_time_stamp()}"
        else:
            run_name = f"{task_type}_{model_type}_{get_time_stamp()}"

    if flags.FLAGS.log_dir:
        logging.get_absl_handler().use_absl_log_file(run_name, flags.FLAGS.log_dir)

    logging.info("========== Config Paths ==========\n" + pformat(flags.FLAGS.config_paths))
    logging.info("========== Config Dict ===========\n" + pformat(config))
    logging.info(f"Run name:\n{run_name}")

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
    save_checkpoints = config["save_checkpoints"]
    restore_checkpoints = config["restore_checkpoints"]
    checkpoint_freq = config["checkpoint_freq"]
    save_best = config["save_best"]
    available_devices = config.get("available_devices")
    model_folder = config["model_folder"]
    do_train = config["train"]
    do_test = config["test"]
    test_with_best = config["test_with_best"]

    output_db_path = config.get("output_db_path", None)
    unique_output_db = config.get("unique_output_db", False)
    assert (output_db_path is None) or (not unique_output_db)

    if unique_output_db:
        output_db_path = f"{run_name}_output_db.json"

    model_dir = os.path.join(model_folder, run_name)

    gpu_devices, n_devices = get_devices(available_devices)
    logging.info(f"GPU devices: {gpu_devices}")

    if batch_size % n_devices > 0:
        raise ValueError('Batch size must be divisible by the number of devices')

    input_shapes, loss_targ_fn, logi_targ_fn = get_task_fns_and_shape(task_type, batch_size,
                                                                      max_len)

    train_ds, eval_ds, test_ds, encoder = get_datasets_and_encoder(
        task_type=task_type, n_devices=n_devices, max_len=max_len, data_kwargs=data_kwargs
    )

    train_ds = train_ds.repeat()
    vocab_size = encoder.vocab_size
    logging.info(f"Vocab Size: {vocab_size}")

    model_kwargs.update({
        'vocab_size': vocab_size,
        'classifier': True,
    })
    logging.info("======= Final Model Kwargs =======\n" + pformat(model_kwargs))

    if flags.FLAGS.dry_run:
        logging.info("Dry run finished...")
        return

    rng = random.PRNGKey(random_seed)
    rng = jax.random.fold_in(rng, jax.process_index())
    rng, init_rng = random.split(rng)
    # We init the first set of train PRNG keys, but update it afterwards inside
    # the main pmap'd training update for performance.
    dropout_rngs = random.split(rng, n_devices)

    tx = train_utils.create_optimiser(**optim_kwargs)
    t_state = train_utils.get_model(model_type, model_base, model_kwargs, init_rng, input_shapes,
                                    tx)

    start_step = 0
    if restore_checkpoints or do_test:
        logging.info(f"Attempting to restore model from checkpoint at {model_dir}")
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

    # Initialise history dict
    history = {
        "train": {k : [] for k in ['learning_rate', 'perplexity', 'loss', 'accuracy',
                                   'steps_per_second']},
        "validation": {k: [] for k in ['perplexity', 'loss', 'accuracy']},
        "testing": {k: [] for k in ['perplexity', 'loss', 'accuracy']},
    }

    if do_train:
        logging.info('======= Starting training ========')
        t_state, _, history = train_utils.train(
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
            history=history,
            checkpoint_freq=checkpoint_freq,
            save_checkpoints=save_checkpoints,
            model_dir=model_dir,
            eval_freq=eval_freq,
            save_best=save_best,
        )

    if do_test:
        logging.info('============ Testing =============')

        if test_with_best:
            logging.info("Loading best model for test")
            t_state = checkpoints.restore_checkpoint(model_dir+"_best", t_state)
            t_state = jax_utils.replicate(t_state, devices=gpu_devices)

        history = train_utils.test(
            test_ds=test_ds,
            n_devices=n_devices,
            p_eval_step=p_eval_step,
            t_state=t_state,
            history=history,
        )

    if output_db_path is not None:
        write_to_output_db(output_db_path=output_db_path, run_name=run_name, model_dir=model_dir,
                           config=config, history=history)


if __name__ == "__main__":
    app.run(main)

