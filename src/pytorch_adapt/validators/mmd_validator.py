from ..layers import MMDBatchedLoss
from ..utils import common_functions as c_f
from .base_validator import BaseValidator


class MMDValidator(BaseValidator):
    def __init__(self, layer="features", batch_size=1024, mmd_kwargs=None, **kwargs):
        super().__init__(**kwargs)
        self.layer = layer
        mmd_kwargs = c_f.default(mmd_kwargs, {})
        self.loss_fn = MMDBatchedLoss(batch_size=batch_size, **mmd_kwargs)

    def compute_score(self, src_train, target_train):
        x = src_train[self.layer]
        y = target_train[self.layer]
        return -self.loss_fn(x, y).item()
