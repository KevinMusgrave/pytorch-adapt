from typing import Callable

import torch
import torch.nn.functional as F

from ..utils import common_functions as c_f


def mcd_loss(out1, out2, dist_fn):
    return dist_fn(F.softmax(out1, dim=1), F.softmax(out2, dim=1))


class MCDLoss(torch.nn.Module):
    """
    Implementation of the loss function used in
    [Maximum Classifier Discrepancy for Unsupervised Domain Adaptation](https://arxiv.org/abs/1712.02560).
    """

    def __init__(self, dist_fn: Callable[[torch.Tensor], torch.Tensor] = None):
        """
        Arguments:
            dist_fn: Computes the mean distance between two softmaxed tensors.
                If ```None```, then ```torch.nn.L1Loss``` is used.
        """
        super().__init__()
        self.dist_fn = c_f.default(dist_fn, torch.nn.L1Loss, {})

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Arguments:
            x: a batch of class logits
            y: the other batch of class logits
        Returns:
            The discrepancy between the two batches of class logits.
        """
        return mcd_loss(x, y, self.dist_fn)


def general_mcd_loss(*x, p=1):
    x = [torch.nn.functional.softmax(i, dim=1) for i in x]
    loss = 0
    for i in range(len(x)):
        for j in range(i + 1, len(x)):
            loss += F(x[i], x[j])
    return loss


class GeneralMCDLoss(torch.nn.Module):
    def __init__(self, p=1):
        super().__init__()
        self.p = p

    def forward(self, *x):
        """"""
        return general_mcd_loss(*x, p=self.p)
