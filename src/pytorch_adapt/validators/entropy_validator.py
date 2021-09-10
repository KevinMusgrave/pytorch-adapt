from ..layers import EntropyLoss
from .base_validator import BaseValidator


class EntropyValidator(BaseValidator):
    """
    Returns the negative of the
    [entropy][pytorch_adapt.layers.entropy_loss.EntropyLoss]
    of all logits.
    """

    def compute_score(self, target_train):
        return -EntropyLoss()(target_train["logits"]).item()
