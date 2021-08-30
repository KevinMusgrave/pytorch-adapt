import torch

from ..utils import common_functions as c_f
from .normalizers import NoNormalizer


class ConfidenceWeights(torch.nn.Module):
    def __init__(self, normalizer=None):
        super().__init__()
        self.normalizer = c_f.default(normalizer, NoNormalizer())

    def forward(self, logits):
        return self.normalizer(torch.max(logits, dim=1)[0])
