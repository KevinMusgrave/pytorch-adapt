from typing import Callable

import torch

from ..utils import common_functions as c_f
from .normalizers import NoNormalizer


class ConfidenceWeights(torch.nn.Module):
    """
    Returns the max value along each row of the input,
    followed by an optional normalization function.
    The output of this can be used to weight
    classification losses by the "confidence" of the predictions.
    """

    def __init__(self, normalizer: Callable[[torch.Tensor], torch.Tensor] = None):
        """
        Arguments:
            normalizer: A callable for normalizing
                (e.g. min-max normalization) the weights.
                If ```None```, then no normalization is used.
        """

        super().__init__()
        self.normalizer = c_f.default(normalizer, NoNormalizer())

    def forward(self, logits):
        """"""
        return self.normalizer(torch.max(logits, dim=1)[0])
