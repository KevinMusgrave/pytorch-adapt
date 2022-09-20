from ..layers import EntropyLoss
from .simple_loss_validator import SimpleLossValidator


class EntropyValidator(SimpleLossValidator):
    """
    Returns the negative of the
    [entropy][pytorch_adapt.layers.entropy_loss.EntropyLoss]
    of all logits.
    """

    @property
    def loss_fn(self):
        return EntropyLoss(after_softmax=self.layer == "preds")
