import torch

from .entropy_loss import entropy


class DiversityLoss(torch.nn.Module):
    def forward(self, logits):
        logits = torch.mean(logits, dim=0, keepdim=True)
        return -torch.mean(entropy(logits))
