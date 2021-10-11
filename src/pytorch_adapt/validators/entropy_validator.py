from ..layers import EntropyLoss
from .base_validator import BaseValidator


class EntropyValidator(BaseValidator):
    """
    Returns the negative of the
    [entropy][pytorch_adapt.layers.entropy_loss.EntropyLoss]
    of all logits.
    """

    def __init__(self, layer="logits", **kwargs):
        super().__init__(**kwargs)
        self.layer = layer
        self.loss_fn = EntropyLoss(after_softmax=self.layer == "preds")

    def compute_score(self, target_train):
        return -self.loss_fn(target_train[self.layer]).item()
