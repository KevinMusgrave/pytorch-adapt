import torch

from ..layers import DiversityLoss
from .base_validator import BaseValidator


class DiversityValidator(BaseValidator):
    def compute_score(self, target_train):
        return DiversityLoss()(target_train["logits"]).item()

    @property
    def maximize(self):
        return False
