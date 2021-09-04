import torch


class AbsLoss(torch.nn.Module):
    """
    The mean absolute value.
    """

    def forward(self, x):
        """"""
        return torch.mean(torch.abs(x))
