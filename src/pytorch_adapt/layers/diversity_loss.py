import torch

from ..utils import common_functions as c_f
from .entropy_loss import get_entropy


class DiversityLoss(torch.nn.Module):
    """
    Encourages predictions to be uniform, batch wise.
    Takes logits (before softmax) as input.

    For example:

    - A tensor with a large loss: ```torch.tensor([[1e4, 0, 0], [1e4, 0, 0], [1e4, 0, 0]])```

    - A tensor with a small loss: ```torch.tensor([[1e4, 0, 0], [0, 1e4, 0], [0, 0, 1e4]])```
    """

    def __init__(self, after_softmax: bool = False):
        """
        Arguments:
            after_softmax: If ```True```, then the rows of the input are assumed to
                already have softmax applied to them.
        """
        super().__init__()
        self.after_softmax = after_softmax

    def forward(self, logits):
        """"""
        logits = torch.mean(logits, dim=0, keepdim=True)
        return -torch.mean(get_entropy(logits, self.after_softmax))

    def extra_repr(self):
        """"""
        return c_f.extra_repr(self, ["after_softmax"])
