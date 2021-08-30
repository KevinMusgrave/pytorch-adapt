import torch


class BNMLoss(torch.nn.Module):
    def forward(self, x):
        x = torch.nn.functional.softmax(x, dim=1)
        return -torch.linalg.norm(x, "nuc") / x.shape[0]
