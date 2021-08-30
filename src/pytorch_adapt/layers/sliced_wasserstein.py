import torch

from ..utils import common_functions as c_f


class SlicedWasserstein(torch.nn.Module):
    def __init__(self, m=128):
        super().__init__()
        self.m = 128

    def forward(self, x, y):
        d = x.shape[1]
        proj = torch.randn(d, self.m, device=x.device)
        proj = torch.nn.functional.normalize(proj, dim=0)
        x = torch.matmul(x, proj)
        y = torch.matmul(y, proj)
        x, _ = torch.sort(x, dim=0)
        y, _ = torch.sort(y, dim=0)
        return torch.mean((x - y) ** 2)

    def extra_repr(self):
        return c_f.extra_repr(self, ["m"])
