import numpy as np

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
