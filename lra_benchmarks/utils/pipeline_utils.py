from lra_benchmarks.utils.train_utils import compute_weighted_cross_entropy
from lra_benchmarks.text_classification import input_pipeline as tc_input_pipeline
from lra_benchmarks.listops import input_pipeline as listops_input_pipeline
from lra_benchmarks.matching import input_pipeline as matching_input_pipeline


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
        loss, weight_sum = compute_weighted_cross_entropy(logits, targets, num_classes=num_classes,
                                                          weights=None)
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


def get_datasets_and_encoder(task_type, n_devices, max_len, data_kwargs):
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
    else:
        raise ValueError(f"Task type {task_type} is unsupported")

    return train_ds, eval_ds, test_ds, encoder


def get_task_fns_and_shape(task_type, batch_size, max_len):
    input_shape = (batch_size, max_len)

    # Use defaults in train_utils unless doing matching task
    if task_type == "matching":
        loss_targ_fn = matching_task_get_loss_fn_and_targets
        logi_targ_fn = matching_task_get_logits_and_targets
        input_shapes = [input_shape, input_shape]
    else:
        loss_targ_fn = None
        logi_targ_fn = None
        input_shapes = [input_shape]

    return input_shapes, loss_targ_fn, logi_targ_fn

