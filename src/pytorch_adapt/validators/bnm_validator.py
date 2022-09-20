from ..layers import BNMLoss
from .simple_loss_validator import SimpleLossValidator


class BNMValidator(SimpleLossValidator):
    """
    Returns the negative of the
    [BNM loss][pytorch_adapt.layers.bnm_loss.BNMLoss]
    of all logits.
    """

    @property
    def loss_fn(self):
        return BNMLoss()
