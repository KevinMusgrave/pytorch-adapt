import inspect
import json
import os
from abc import ABC, abstractmethod

import numpy as np
from pytorch_metric_learning.utils import common_functions as pml_cf

from ..utils import common_functions as c_f


class BaseValidator(ABC):
    def __init__(self, normalizer=None, key_map=None, ignore_epoch=0):
        self.normalizer = c_f.default(normalizer, return_raw)
        self.score_history = np.array([])
        self.raw_score_history = np.array([])
        self.epochs = np.array([], dtype=int)
        self.key_map = c_f.default(key_map, {})
        self.ignore_epoch = ignore_epoch
        pml_cf.add_to_recordable_attributes(
            self,
            list_of_names=["latest_score", "best_score", "latest_epoch", "best_epoch"],
        )

    def _required_data(self):
        args = inspect.getfullargspec(self.compute_score).args
        args.remove("self")
        return args

    @property
    def required_data(self):
        output = set(self._required_data()) - set(self.key_map.values())
        output = list(output)
        for k, v in self.key_map.items():
            output.append(k)
        return output

    @abstractmethod
    def compute_score(self):
        pass

    def score(self, epoch, **kwargs):
        kwargs = self.kwargs_check(epoch, kwargs)
        score = self.compute_score(**kwargs)
        self.append_to_history_and_normalize(score, epoch)
        return self.latest_score

    def kwargs_check(self, epoch, kwargs):
        if epoch in self.epochs:
            raise ValueError(f"Epoch {epoch} has already been evaluated")
        if kwargs.keys() != set(self.required_data):
            raise ValueError(
                f"Input to compute_score has keys = {kwargs.keys()} but should have keys {self.required_data}"
            )
        return c_f.map_keys(kwargs, self.key_map)

    def append_to_history_and_normalize(self, score, epoch):
        self.raw_score_history = np.append(self.raw_score_history, score)
        self.epochs = np.append(self.epochs, epoch)
        self.score_history = self.normalizer(self.raw_score_history)

    @property
    def score_history_ignore_epoch(self):
        return remove_ignore_epoch(self.score_history, self.epochs, self.ignore_epoch)

    @property
    def epochs_ignore_epoch(self):
        return remove_ignore_epoch(self.epochs, self.epochs, self.ignore_epoch)

    def has_valid_history(self, ignore_ignore_epoch=True):
        x = (
            self.score_history_ignore_epoch
            if ignore_ignore_epoch
            else self.score_history
        )
        return len(x) > 0 and np.isfinite(x).any()

    @property
    def best_score(self):
        if self.has_valid_history():
            return self.score_history_ignore_epoch[self.best_idx]

    @property
    def best_epoch(self):
        if self.has_valid_history():
            return self.epochs_ignore_epoch[self.best_idx]

    @property
    def best_idx(self):
        if self.has_valid_history():
            return (
                np.nanargmax(self.score_history_ignore_epoch)
                if self.maximize
                else np.nanargmin(self.score_history_ignore_epoch)
            )

    @property
    def latest_epoch(self):
        if self.has_valid_history(False):
            return self.epochs[-1]

    @property
    def latest_score(self):
        if self.has_valid_history(False):
            return self.score_history[-1]

    @property
    def latest_is_best(self):
        if self.has_valid_history(False):
            return self.best_epoch == self.latest_epoch
        return False

    @property
    def maximize(self):
        return True

    def __repr__(self):
        return c_f.nice_repr(self, self.extra_repr(), {})

    def extra_repr(self):
        return c_f.extra_repr(
            self, ["latest_score", "best_score", "best_epoch", "required_data"]
        )


def score_dict_to_str(d):
    return "\n\t" + "\n\t".join([f"{k}: {v}" for k, v in d.items()])


def remove_ignore_epoch(x, epochs, ignore_epoch):
    if ignore_epoch is not None:
        return x[epochs != ignore_epoch]
    return x


def return_raw(raw_score_history):
    return raw_score_history
