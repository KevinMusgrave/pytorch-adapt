import numpy as np
import torch

from ..utils import common_functions as c_f
from .base_validator import BaseValidator


def max_normalizer(raw_score_history):
    return raw_score_history / np.abs(np.nanmax(raw_score_history))


def call_val_hook(hook, collected_data, epoch=None):
    kwargs = collected_data
    if hasattr(hook, "required_data"):
        kwargs = c_f.filter_kwargs(collected_data, hook.required_data)
    if isinstance(hook, BaseValidator):
        return hook(**kwargs)
    else:
        return hook(epoch, **kwargs)


##########################
# used by PerClassValidator and TargetKNNValidator


def src_label_fn(x):
    return x["labels"]


def target_label_fn(x):
    return torch.argmax(x["logits"], dim=1)


def default_label_fns(required_data):
    output = {}
    for k in required_data:
        if k.startswith("src"):
            label_fn = src_label_fn
        elif k.startswith("target"):
            label_fn = target_label_fn
        else:
            raise ValueError(f"expected key to start with 'src' or 'target but got {k}")
        output[k] = label_fn
    return output


##########################
