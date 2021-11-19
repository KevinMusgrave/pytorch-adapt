from ..layers import DiversityLoss
from .base_validator import BaseValidator


class DiversityValidator(BaseValidator):
    """
    Returns the negative of the
    [diversity][pytorch_adapt.layers.diversity_loss.DiversityLoss]
    of all logits.
    """

    def __init__(self, layer="logits", **kwargs):
        super().__init__(**kwargs)
        self.layer = layer
        self.loss_fn = DiversityLoss(after_softmax=self.layer == "preds")

    def compute_score(self, target_train):
        return -self.loss_fn(target_train[self.layer]).item()
