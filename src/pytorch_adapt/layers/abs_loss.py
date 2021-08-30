import torch


class AbsLoss(torch.nn.Module):
    def forward(self, x):
        return torch.mean(torch.abs(x))
