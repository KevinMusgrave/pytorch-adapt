import torch
from pytorch_metric_learning.utils import common_functions as pml_cf

from ..utils import common_functions as c_f
from .concat_softmax import ConcatSoftmax
from .utils import split_half


class SymNetsDomainLoss(torch.nn.Module):
    def __init__(self, half_idx):
        super().__init__()
        self.half_idx = half_idx
        self.softmax_fn = ConcatSoftmax()

    # x and y are the first and second halves of "p^st"
    def forward(self, x, y):
        x = self.softmax_fn(x, y)
        x = split_half(x, dim=1)[self.half_idx]
        x = torch.sum(x, dim=1)
        x[x == 0] += pml_cf.small_val(x.dtype)
        return -torch.mean(torch.log(x))

    def extra_repr(self):
        return c_f.extra_repr(self, ["half_idx"])
