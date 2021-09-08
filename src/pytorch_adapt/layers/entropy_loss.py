import torch
from pytorch_metric_learning.utils import common_functions as pml_cf

from ..utils import common_functions as c_f


def entropy(logits):
    entropies = -torch.sum(
        torch.softmax(logits, dim=1) * torch.log_softmax(logits, dim=1), dim=1
    )
    return entropies


def entropy_after_softmax(preds, eps=None):
    eps = c_f.default(eps, pml_cf.small_val(preds.dtype))
    entropies = -torch.sum(preds * torch.log(preds + eps), dim=1)
    return entropies


def get_entropy(logits, after_softmax):
    if after_softmax:
        return entropy_after_softmax(logits)
    return entropy(logits)


class EntropyLoss(torch.nn.Module):
    """
    Encourages low entropy predictions, or in other words, "confident" predictions.
    """

    def __init__(self, after_softmax: bool = False, return_mean: bool = True):
        """
        Arguments:
            after_softmax: If ```True```, then the rows of the input are assumed to
                already have softmax applied to them.
            return_mean: If ```True```, the mean entropy will be returned.
                If ```False```, the entropy per row of the input will be returned.
        """
        super().__init__()
        self.after_softmax = after_softmax
        self.return_mean = return_mean

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Arguments:
            logits: Raw logits if ```self.after_softmax``` is False.
                Otherwise each row should be predictions that sum up to 1.
        """
        entropies = get_entropy(logits, self.after_softmax)
        if self.return_mean:
            return torch.mean(entropies)
        return entropies

    def extra_repr(self):
        """"""
        return c_f.extra_repr(self, ["after_softmax"])
