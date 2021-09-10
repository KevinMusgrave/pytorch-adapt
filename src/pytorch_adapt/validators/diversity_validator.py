import torch

from ..layers import DiversityLoss
from .base_validator import BaseValidator


class DiversityValidator(BaseValidator):
    """
    Returns the negative of the
    [diversity][pytorch_adapt.layers.diversity_loss.DiversityLoss]
    of all logits.
    """

    def compute_score(self, target_train):
        return -DiversityLoss()(target_train["logits"]).item()
