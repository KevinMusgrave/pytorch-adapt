from abc import ABC
from typing import Callable, Dict

import numpy as np
import torch
from pytorch_metric_learning.utils import common_functions as pml_cf

from ..utils import common_functions as c_f
from .multiple_validators import MultipleValidators


class ScoreHistory(ABC):
    """
    The parent class of all validators.

    The main purpose of validators is to give an estimate
    of target domain accuracy, without having access to
    class labels.
    """

    def __init__(
        self,
        validator,
        normalizer: Callable[[np.ndarray], np.ndarray] = None,
        ignore_epoch: int = None,
    ):
        """
        Arguments:
            normalizer: A function that receives the current unnormalized
                score history, and returns a normalized version of the
                score history. If ```None```, then it defaults to
                no normalization.
            ignore_epoch: This epoch will ignored when determining
                the best scoring epoch. The default of 0 is meant to be
                reserved for the initial model (before training has begun).
        """
        self.validator = validator
        self.normalizer = c_f.default(normalizer, return_raw)
        self.score_history = np.array([])
        self.raw_score_history = np.array([])
        self.epochs = np.array([], dtype=int)
        self.ignore_epoch = ignore_epoch
        pml_cf.add_to_recordable_attributes(
            self,
            list_of_names=["latest_score", "best_score", "latest_epoch", "best_epoch"],
        )

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
        if epoch in self.epochs:
            raise ValueError(f"Epoch {epoch} has already been evaluated")
        score = self.validator.score(**kwargs)
        sub_scores = None
        if isinstance(score, (list, tuple)):
            score, sub_scores = score
        self.append_to_history_and_normalize(score, epoch)
        if sub_scores:
            return self.latest_score, sub_scores
        return self.latest_score

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
            return np.nanargmax(self.score_history_ignore_epoch)

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
    def required_data(self):
        return self.validator.required_data

    def __repr__(self):
        return c_f.nice_repr(self, self.extra_repr(), {})

    def extra_repr(self):
        return c_f.extra_repr(
            self, ["validator", "latest_score", "best_score", "best_epoch"]
        )


class ScoreHistories(ScoreHistory):
    def __init__(self, validator, **kwargs):
        super().__init__(validator=validator, **kwargs)
        if not isinstance(validator, MultipleValidators):
            raise TypeError("validator must be of type MultipleValidators")
        validator.return_sub_scores = True
        self.histories = {k: ScoreHistory(v) for k, v in validator.validators.items()}
        pml_cf.add_to_recordable_attributes(self, list_of_names=["histories"])

    def score(self, epoch: int, **kwargs: Dict[str, torch.Tensor]) -> float:
        score, sub_scores = super().score(epoch, **kwargs)
        for k, v in self.histories.items():
            v.append_to_history_and_normalize(sub_scores[k], epoch)
        return score

    def extra_repr(self):
        x = super().extra_repr()
        x += f"\n{c_f.extra_repr(self, ['histories'])}"
        return x


def remove_ignore_epoch(x, epochs, ignore_epoch):
    if ignore_epoch is not None:
        return x[epochs != ignore_epoch]
    return x


def return_raw(raw_score_history):
    return raw_score_history
