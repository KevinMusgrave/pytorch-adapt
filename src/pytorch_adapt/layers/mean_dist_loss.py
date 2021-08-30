import torch

from ..utils import common_functions as c_f


class MeanDistLoss(torch.nn.Module):
    def __init__(self, p=2):
        super().__init__()
        self.p = p

    def forward(self, x, y):
        return torch.mean(torch.cdist(x, y, p=self.p))

    def extra_repr(self):
        return c_f.extra_repr(self, ["p"])
