from ..layers import DiversityLoss
from .simple_loss_validator import SimpleLossValidator


class DiversityValidator(SimpleLossValidator):
    """
    Returns the negative of the
    [diversity][pytorch_adapt.layers.diversity_loss.DiversityLoss]
    of all logits.
    """

    @property
    def loss_fn(self):
        return DiversityLoss(after_softmax=self.layer == "preds")
