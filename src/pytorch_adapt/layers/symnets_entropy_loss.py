import torch

from .concat_softmax import ConcatSoftmax
from .entropy_loss import EntropyLoss
from .utils import split_half


class SymNetsEntropyLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.softmax_fn = ConcatSoftmax()
        self.entropy_loss_fn = EntropyLoss(after_softmax=True)

    def forward(self, x, y):
        x = self.softmax_fn(x, y)
        x, y = split_half(x, dim=1)
        return self.entropy_loss_fn(x + y)


class SymNetsEntropyLossListInput(SymNetsEntropyLoss):
    def forward(self, x):
        return super().forward(*x)
