import torch

from ..utils import common_functions as c_f


def batch_spectral_loss(x, k):
    singular_values = torch.linalg.svdvals(x)
    return torch.sum(singular_values[:k] ** 2)


class BatchSpectralLoss(torch.nn.Module):
    """
    Implementation of the loss in
    [Transferability vs. Discriminability: Batch Spectral
    Penalization for Adversarial Domain Adaptation](http://proceedings.mlr.press/v97/chen19i.html).
    The loss is the sum of the squares of the first k singular values.
    """

    def __init__(self, k: int = 1):
        """
        Arguments:
            k: the number of singular values to include in the loss
        """
        super().__init__()
        self.k = k

    def forward(self, x):
        """"""
        return batch_spectral_loss(x, self.k)

    def extra_repr(self):
        """"""
        return c_f.extra_repr(self, ["k"])
