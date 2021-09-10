import torch

from ..utils import common_functions as c_f


def sum_normalizer(x, detach=False, scale_by_batch_size=False):
    y = torch.sum(x)
    if detach:
        y = y.detach()
    if scale_by_batch_size:
        x = x * (x.shape[0])
    return x / y


def min_max_normalizer(x, detach=False):
    x_min = torch.min(x)
    x_max = torch.max(x)
    if detach:
        x_min = x_min.detach()
        x_max = x_max.detach()
    return (x - x_min) / (x_max - x_min)


def max_normalizer(x, detach=False):
    x_max = torch.max(x)
    if detach:
        x_max = x_max.detach()
    return x / x_max


def no_normalizer(x):
    return x


class SumNormalizer(torch.nn.Module):
    def __init__(self, detach=False, scale_by_batch_size=False):
        super().__init__()
        self.detach = detach
        self.scale_by_batch_size = scale_by_batch_size

    def forward(self, x):
        return sum_normalizer(
            x, detach=self.detach, scale_by_batch_size=self.scale_by_batch_size
        )

    def extra_repr(self):
        return c_f.extra_repr(self, ["detach", "scale_by_batch_size"])


class MinMaxNormalizer(torch.nn.Module):
    def __init__(self, detach=False):
        super().__init__()
        self.detach = detach

    def forward(self, x):
        return min_max_normalizer(x, detach=self.detach)

    def extra_repr(self):
        return c_f.extra_repr(self, ["detach"])


class MaxNormalizer(torch.nn.Module):
    def __init__(self, detach=False):
        super().__init__()
        self.detach = detach

    def forward(self, x):
        return max_normalizer(x, detach=self.detach)

    def extra_repr(self):
        return c_f.extra_repr(self, ["detach"])


class NoNormalizer(torch.nn.Module):
    def forward(self, x):
        return no_normalizer(x)
