from typing import Callable

import torch

from ..utils import common_functions as c_f
from .entropy_weights import EntropyWeights
from .normalizers import SumNormalizer


# reference https://github.com/thuml/Versatile-Domain-Adaptation
class MCCLoss(torch.nn.Module):
    """
    Implementation of
    [Minimum Class Confusion for Versatile Domain Adaptation](https://arxiv.org/abs/1912.03699).
    """

    def __init__(
        self,
        T: float = 1,
        entropy_weighter: Callable[[torch.Tensor], torch.Tensor] = None,
    ):
        """
        Arguments:
            T: softmax temperature applied to the input target logits
            entropy_weighter: a function that returns a weight for each
                sample. The weights are used in the process of computing
                the class confusion tensor as described in the paper.
                If ```None```, then ```layers.EntropyWeights``` is used.
        """
        super().__init__()
        self.T = T
        self.entropy_weighter = c_f.default(
            entropy_weighter,
            EntropyWeights(
                after_softmax=True, normalizer=SumNormalizer(scale_by_batch_size=True)
            ),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Arguments:
            x: target logits
        """
        Y = torch.nn.functional.softmax(x / self.T, dim=1)
        H_weights = self.entropy_weighter(Y.detach())
        C = torch.linalg.multi_dot([Y.t(), torch.diag(H_weights), Y])
        C = C / torch.sum(C, dim=1)
        return (torch.sum(C) - torch.trace(C)) / C.shape[0]

    def extra_repr(self):
        """"""
        return c_f.extra_repr(self, ["T"])
