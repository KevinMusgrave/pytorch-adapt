import numpy as np

from ..layers import MMDLoss
from ..utils import common_functions as c_f
from .base_validator import BaseValidator


def randomly_sample(x, num_samples):
    if num_samples == len(x):
        return x
    return x[np.random.choice(len(x), size=num_samples, replace=True)]


class MMDValidator(BaseValidator):
    def __init__(
        self,
        layer="features",
        num_samples="max",
        num_trials=1,
        mmd_kwargs=None,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.layer = layer
        if not isinstance(num_samples, int) and num_samples != "max":
            raise ValueError("num_samples must be an int or 'max'")
        self.num_samples = num_samples
        self.num_trials = num_trials
        mmd_kwargs = c_f.default(mmd_kwargs, {})
        self.loss_fn = MMDLoss(**mmd_kwargs)

    def compute_score(self, src_train, target_train):
        x = src_train[self.layer]
        y = target_train[self.layer]
        score = []
        num_samples = (
            np.max([len(x), len(y)]) if self.num_samples == "max" else self.num_samples
        )
        for i in range(self.num_trials):
            curr_x = randomly_sample(x, num_samples)
            curr_y = randomly_sample(y, num_samples)
            score.append(self.loss_fn(curr_x, curr_y).item())
        return -np.mean(score)
