import torch
import torch.nn.functional as F

from .concat_softmax import ConcatSoftmax
from .utils import split_half


class SymNetsCategoryLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.softmax_fn = ConcatSoftmax()

    # x and y are the first and second halves of "p^st"
    def forward(self, x, y, src_labels):
        x = self.softmax_fn(x, y)
        x, y = split_half(x, dim=1)
        x_loss = F.cross_entropy(x, src_labels)
        y_loss = F.cross_entropy(y, src_labels)
        return x_loss + y_loss


class SymNetsCategoryLossListInput(SymNetsCategoryLoss):
    def forward(self, x, src_labels):
        return super().forward(*x, src_labels)
