import torch

from ..utils import common_functions as c_f


class ConcatSoftmax(torch.nn.Module):
    def __init__(self, dim=1):
        super().__init__()
        self.dim = dim

    def forward(self, *x):
        all_logits = torch.cat(x, dim=self.dim)
        return torch.nn.functional.softmax(all_logits, dim=self.dim)

    def extra_repr(self):
        return c_f.extra_repr(self, ["dim"])
