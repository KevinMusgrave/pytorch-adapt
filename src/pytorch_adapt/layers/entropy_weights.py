import torch

from ..utils import common_functions as c_f
from .entropy_loss import get_entropy
from .normalizers import sum_normalizer


def entropy_weights(logits, after_softmax, normalizer):
    entropies = get_entropy(logits, after_softmax)
    weights = 1 + torch.exp(-entropies)
    return normalizer(weights)


class EntropyWeights(torch.nn.Module):
    def __init__(self, after_softmax=False, normalizer=None):
        super().__init__()
        self.after_softmax = after_softmax
        self.normalizer = c_f.default(normalizer, sum_normalizer)

    def forward(self, logits):
        return entropy_weights(logits, self.after_softmax, self.normalizer)

    def extra_repr(self):
        return c_f.extra_repr(self, ["after_softmax"])
