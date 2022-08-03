from typing import Callable, Any
from copy import deepcopy
import json
from absl import logging

from flax.core import FrozenDict


def txt_str_to_md(string, bullet_consecutive_lines=False):

    # Make things titles based on newlines
    string = '\n\n\n## '.join(string.split('\n\n\n'))

    # If applicable, make lists out of consective, non-empty lines
    if bullet_consecutive_lines:
        s_split = string.split('\n')
        new_string = [s_split[0], s_split[1]]

        # We want to bullet when the previous line is not empty and neither is the current
        # If we're at the top of the bullets we need to insert a newline
        for l1, l2, l3 in zip(s_split, s_split[1:], s_split[2:]):
            if l2 != '' and l3 != '':
                if l1 == '':
                    new_string.append("\n* " + l3)
                else:
                    new_string.append("* " + l3)
            else:
                new_string.append(l3)

        string = '\n'.join(new_string)

    # Insert header and make first line title
    string = "---\ngeometry: margin=2.5cm\n---\n\n# " + string

    return string


def ensure_suffix(txt: str, suffix: str) -> str:
    return txt if txt[-len(suffix):] == suffix else txt + suffix


def ensure_prefix(prefix: str, txt: str) -> str:
    return txt if txt[:len(prefix)] == prefix else prefix + txt


def eval_fn_with_maybe_kwargs(fn: Callable, *args, kwarg_dict: dict[str, Any], keys: list[str],
                              **kwargs):
    fn_kwargs = {k: kwarg_dict[k] for k in keys if k in kwarg_dict.keys()}
    return fn(*args, **fn_kwargs, **kwargs)


def is_dict(x):
    return isinstance(x, dict) or isinstance(x, FrozenDict)


def recursive_dict_update(base: dict, target: dict, assert_type_match=True,
                          type_match_ignore_nones=True, extend_lists=False) -> dict:
    new = deepcopy(base)

    for k, v in target.items():

        if assert_type_match and (k in base.keys()):
            t1, t2 = type(base[k]), type(v)
            if t1 != t2 and (not type_match_ignore_nones or (t1 is None or t2 is None)):
                raise ValueError(f"Types do not match for key {k}, {t1} vs {t2}")

        if (k not in base.keys()):
            new[k] = v
        elif is_dict(v) and is_dict(base[k]):
            new[k] = recursive_dict_update(base[k], v, assert_type_match=assert_type_match,
                                           type_match_ignore_nones=type_match_ignore_nones,
                                           extend_lists=extend_lists)
        elif isinstance(v, list) and isinstance(base[k], list) and extend_lists:
            new[k] += v
        else:
            new[k] = v

    return new


def write_to_output_db(output_db_path, run_name, model_dir, config, history):
    logging.info("Saving metrics and config data...")

    try:
        with open(output_db_path) as f:
            output_db = json.load(f)

    except FileNotFoundError:
        logging.warning("Existing output db does not exist or was not found")
        output_db = {}

    if run_name in output_db.keys():
        logging.info("Found existing run, updating entry")
        entry = output_db[run_name]

        new_history = recursive_dict_update(entry["history"], history, extend_lists=True)

        if isinstance(entry["config"], list):
            configs = entry["config"]
        else:
            configs = [entry["config"]]

        configs.append(config)

        output_db[run_name] = {
            "model_dir": model_dir,
            "config": configs,
            "history": new_history}

    else:
        output_db[run_name] = {
            "model_dir": model_dir,
            "config": config,
            "history": history}

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
        output_db[run_name] = None

        with open(output_db_path, 'w', encoding="utf-8") as f:
            json.dump(output_db, f, ensure_ascii=False, indent=4)

        raise json_exception

