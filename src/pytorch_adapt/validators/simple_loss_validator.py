from abc import abstractmethod

from .base_validator import BaseValidator


class SimpleLossValidator(BaseValidator):
    def __init__(self, layer="logits", **kwargs):
        super().__init__(**kwargs)
        self.layer = layer

    def compute_score(self, target_train):
        return -self.loss_fn(target_train[self.layer]).item()

    @property
    @abstractmethod
    def loss_fn(self, *args, **kwargs):
        pass
