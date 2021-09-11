import inspect
import json
import os
from abc import ABC, abstractmethod
from typing import Callable, Dict, List

import numpy as np
import torch
from pytorch_metric_learning.utils import common_functions as pml_cf

from ..utils import common_functions as c_f


class BaseValidator(ABC):
    """
    The parent class of all validators.

    The main purpose of validators is to give an estimate
    of target domain accuracy, without having access to
    class labels.
    """

    def __init__(
        self,
        normalizer: Callable[[np.ndarray], np.ndarray] = None,
        key_map: Dict[str, str] = None,
        ignore_epoch: int = 0,
    ):
        """
        Arguments:
            normalizer: A function that receives the current unnormalized
                score history, and returns a normalized version of the
                score history. If ```None```, then it defaults to
                no normalization.
            key_map: A mapping from ```<new_split_names>``` to
                ```<original_split_names>```. For example,
                [```AccuracyValidator```][pytorch_adapt.validators.AccuracyValidator]
                expects ```src_val``` by default. When used with one of the
                [```frameworks```](../frameworks/index.md), this default
                indicates that data related to the ```src_val``` split should be retrieved.
                If you instead want to compute accuracy for the ```src_train``` split,
                you would set the ```key_map``` to ```{"src_train": "src_val"}```.
            ignore_epoch: This epoch will ignored when determining
                the best scoring epoch. The default of 0 is meant to be
                reserved for the initial model (before training has begun).
        """
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
    def required_data(self) -> List[str]:
        """
        Returns:
            A list of dataset split names.
        """
        output = set(self._required_data()) - set(self.key_map.values())
        output = list(output)
        for k, v in self.key_map.items():
            output.append(k)
        return output

    @abstractmethod
    def compute_score(self):
        pass

    def score(self, epoch: int, **kwargs: Dict[str, torch.Tensor]) -> float:
        """
        Arguments:
            epoch: The epoch to be scored.
            **kwargs: A mapping from dataset split name to
                dictionaries containing:

                - ```"features"```
                - ```"logits"```
                - ```"preds"```
                - ```"domain"```
                - ```"src_labels"``` (if available)
        """
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
    def best_score(self) -> float:
        """
        Returns:
            The best score, ignoring ```self.ignore_epoch```.
            Returns ```None``` if no valid scores are available.
        """
        if self.has_valid_history():
            return self.score_history_ignore_epoch[self.best_idx]

    @property
    def best_epoch(self) -> int:
        """
        Returns:
            The best epoch, ignoring ```self.ignore_epoch```.
            Returns ```None``` if no valid epochs are available.
        """
        if self.has_valid_history():
            return self.epochs_ignore_epoch[self.best_idx]

    @property
    def best_idx(self) -> int:
        """
        Returns:
            The index of the best score in ```self.score_history_ignore_epoch```.
            Returns ```None``` if no valid epochs are available.
        """
        if self.has_valid_history():
            return (
                np.nanargmax(self.score_history_ignore_epoch)
                if self.maximize
                else np.nanargmin(self.score_history_ignore_epoch)
            )

    @property
    def latest_epoch(self) -> int:
        """
        Returns:
            The latest epoch, including ```self.ignore_epoch```.
            Returns ```None``` if no epochs have been scored.
        """
        if self.has_valid_history(False):
            return self.epochs[-1]

    @property
    def latest_score(self) -> float:
        """
        Returns:
            The latest score, including ```self.ignore_epoch```.
            Returns ```None``` if no epochs have been scored.
        """
        if self.has_valid_history(False):
            return self.score_history[-1]

    @property
    def latest_is_best(self) -> bool:
        """
        Returns:
            ```False``` if the latest epoch was not the best
            scoring epoch, or if the latest epoch is ```ignore_epoch```.
            ```True``` otherwise.
        """
        if self.has_valid_history(False):
            return self.best_epoch == self.latest_epoch
        return False

    @property
    def maximize(self) -> bool:
        """
        Returns:
            ```True``` if a higher validation score indicates a better model.
            The default is ```True```.
        """
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
