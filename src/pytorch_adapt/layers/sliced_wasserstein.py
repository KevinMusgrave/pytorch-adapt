import torch

from ..utils import common_functions as c_f


class SlicedWasserstein(torch.nn.Module):
    """
    Implementation of the loss used in
    [Sliced Wasserstein Discrepancy for Unsupervised Domain Adaptation](https://arxiv.org/abs/1903.04064)
    """

    def __init__(self, m: int = 128):
        """
        Arguments:
            m: The dimensionality to project to.
        """
        super().__init__()
        self.m = 128

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Arguments:
            x: a batch of class predictions
            y: the other batch of class predictions
        Returns:
            The discrepancy between the two batches of class predictions.
        """
        d = x.shape[1]
        proj = torch.randn(d, self.m, device=x.device)
        proj = torch.nn.functional.normalize(proj, dim=0)
        x = torch.matmul(x, proj)
        y = torch.matmul(y, proj)
        x, _ = torch.sort(x, dim=0)
        y, _ = torch.sort(y, dim=0)
        return torch.mean((x - y) ** 2)

    def extra_repr(self):
        """"""
        return c_f.extra_repr(self, ["m"])
