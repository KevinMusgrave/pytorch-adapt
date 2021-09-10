from typing import Callable

import torch

from ..utils import common_functions as c_f
from .entropy_loss import get_entropy
from .normalizers import SumNormalizer


def entropy_weights(logits, after_softmax, normalizer):
    entropies = get_entropy(logits, after_softmax)
    weights = 1 + torch.exp(-entropies)
    return normalizer(weights)


class EntropyWeights(torch.nn.Module):
    """
    Implementation of entropy weighting described in
    [Conditional Adversarial Domain Adaptation](https://arxiv.org/abs/1705.10667).
    Computes the entropy (```x```) per row of the input, and returns
    ```1+exp(-x)```.
    This can be used to weight losses, such that the most
    confidently scored samples have a higher weighting.
    """

    def __init__(
        self,
        after_softmax: bool = False,
        normalizer: Callable[[torch.Tensor], torch.Tensor] = None,
    ):
        """
        Arguments:
            after_softmax: If ```True```, then the rows of the input are assumed to
                already have softmax applied to them.
            normalizer: A callable for normalizing
                (e.g. min-max normalization) the weights.
                If ```None```, then sum normalization is used.
        """
        super().__init__()
        self.after_softmax = after_softmax
        self.normalizer = c_f.default(normalizer, SumNormalizer, {})

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Arguments:
            logits: Raw logits if ```self.after_softmax``` is False.
                Otherwise each row should be predictions that sum up to 1.
        """
        return entropy_weights(logits, self.after_softmax, self.normalizer)

    def extra_repr(self):
        """"""
        return c_f.extra_repr(self, ["after_softmax"])
