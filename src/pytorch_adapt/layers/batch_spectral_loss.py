import torch

from ..utils import common_functions as c_f


def batch_spectral_loss(x, k):
    singular_values = torch.linalg.svdvals(x)
    return torch.sum(singular_values[:k] ** 2)


class BatchSpectralLoss(torch.nn.Module):
    # k is the number of singular values to include in the loss
    def __init__(self, k=1):
        super().__init__()
        self.k = k

    def forward(self, x):
        return batch_spectral_loss(x, self.k)

    def extra_repr(self):
        return c_f.extra_repr(self, ["k"])
