from collections import defaultdict

import numpy as np
import torch

from ..utils import common_functions as c_f
from .base_validator import BaseValidator
from .utils import default_label_fns


def check_keys(src_key, target_key):
    if not src_key.startswith("src") or not target_key.startswith("target"):
        raise ValueError(
            f"Expected keys to start with 'src' and 'target', but got {src_key}, {target_key}"
        )


def get_common_labels(kwargs, label_fns):
    labels = {}
    label_sets = []
    for split_name, v1 in kwargs.items():
        curr_labels = label_fns[split_name](v1)
        labels[split_name] = curr_labels
        label_sets.append(set(torch.unique(curr_labels).tolist()))
    common_labels = set.intersection(*label_sets)
    return labels, common_labels


def create_new_kwargs(kwargs, labels, common_labels):
    new_kwargs = []
    for x in common_labels:
        new_kwargs.append(defaultdict(dict))
        for split_name, v1 in kwargs.items():
            curr_labels = labels[split_name]
            for feature_name, v2 in v1.items():
                new_kwargs[-1][split_name][feature_name] = v2[curr_labels == x]
    return new_kwargs


class PerClassValidator(BaseValidator):
    def __init__(self, validator, label_fns=None, **kwargs):
        super().__init__(**kwargs)
        self.validator = validator
        self.label_fns = c_f.default(label_fns, default_label_fns, [self.required_data])
        if self.key_map != {}:
            raise ValueError("key_map should be passed to the wrapped validator")

    def _required_data(self):
        return self.validator.required_data

    def compute_score(self, **kwargs):
        labels, common_labels = get_common_labels(kwargs, self.label_fns)
        new_kwargs = create_new_kwargs(kwargs, labels, common_labels)
        scores = [self.validator(**nk) for nk in new_kwargs]
        return np.mean(scores)

    def extra_repr(self):
        x = super().extra_repr()
        x += f"\n{c_f.extra_repr(self, ['validator'])}"
        return x
