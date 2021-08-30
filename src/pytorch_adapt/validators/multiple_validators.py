import itertools

import numpy as np
from pytorch_metric_learning.utils import common_functions as pml_cf

from ..utils import common_functions as c_f
from .base_validator import BaseValidator


class MultipleValidators(BaseValidator):
    def __init__(self, validators, weights=None, **kwargs):
        super().__init__(**kwargs)
        self.validators = c_f.enumerate_to_dict(validators)
        self.weights = c_f.default(
            weights, {k: 1 if v.maximize else -1 for k, v in self.validators.items()}
        )
        self.weights = c_f.enumerate_to_dict(self.weights)
        pml_cf.add_to_recordable_attributes(
            self, list_of_names=["validators", "weights"]
        )

    def _required_data(self):
        output = [v.required_data for v in self.validators.values()]
        output = list(itertools.chain(*output))
        return list(set(output))

    def compute_score(self):
        pass

    def score(self, epoch, **kwargs):
        kwargs = self.kwargs_check(epoch, kwargs)
        for v in self.validators.values():
            v.score(epoch, **c_f.filter_kwargs(kwargs, v.required_data))
        self.append_to_history_and_normalize(epoch)
        return self.latest_score

    def append_to_history_and_normalize(self, epoch):
        self.raw_score_history = []
        for k, v in self.validators.items():
            self.raw_score_history.append(v.score_history * self.weights[k])
        self.raw_score_history = sum(self.raw_score_history)
        self.epochs = np.append(self.epochs, epoch)
        self.score_history = self.normalizer(self.raw_score_history)

    def __repr__(self):
        return c_f.nice_repr(self, self.extra_repr(), self.validators)

    def extra_repr(self):
        x = super().extra_repr()
        x += f"\n{c_f.extra_repr(self, ['weights'])}"
        return x
