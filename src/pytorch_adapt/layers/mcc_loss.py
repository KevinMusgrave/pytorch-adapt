import torch

from ..utils import common_functions as c_f
from .entropy_loss import entropy_after_softmax
from .entropy_weights import EntropyWeights
from .normalizers import SumNormalizer


# reference https://github.com/thuml/Versatile-Domain-Adaptation
class MCCLoss(torch.nn.Module):
    def __init__(self, T=1, entropy_weighter=None):
        super().__init__()
        self.T = T
        self.entropy_weighter = c_f.default(
            entropy_weighter,
            EntropyWeights(
                after_softmax=True, normalizer=SumNormalizer(scale_by_batch_size=True)
            ),
        )

    def forward(self, x):
        Y = torch.nn.functional.softmax(x / self.T, dim=1)
        H_weights = self.entropy_weighter(Y.detach())
        C = torch.linalg.multi_dot([Y.t(), torch.diag(H_weights), Y])
        C = C / torch.sum(C, dim=1)
        return (torch.sum(C) - torch.trace(C)) / C.shape[0]

    def extra_repr(self):
        return c_f.extra_repr(self, ["T"])
