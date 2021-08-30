import torch
import torch.nn.functional as F

from ..utils import common_functions as c_f


def mcd_loss(out1, out2, dist_fn):
    return dist_fn(F.softmax(out1, dim=1), F.softmax(out2, dim=1))


class MCDLoss(torch.nn.Module):
    def __init__(self, dist_fn=None):
        super().__init__()
        self.dist_fn = c_f.default(dist_fn, torch.nn.L1Loss, {})

    def forward(self, x, y):
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
        return general_mcd_loss(*x, p=self.p)
