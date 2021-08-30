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
    def __init__(self, after_softmax=False, return_mean=True):
        super().__init__()
        self.after_softmax = after_softmax
        self.return_mean = return_mean

    def forward(self, logits):
        entropies = get_entropy(logits, self.after_softmax)
        if self.return_mean:
            return torch.mean(entropies)
        return entropies

    def extra_repr(self):
        return c_f.extra_repr(self, ["after_softmax"])
