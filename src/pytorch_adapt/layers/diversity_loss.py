import torch

from .entropy_loss import entropy


class DiversityLoss(torch.nn.Module):
    """
    Encourages predictions to be uniform, batch wise.
    Takes logits (before softmax) as input.

    For example:

    - A tensor with a large loss: ```torch.tensor([[1e4, 0, 0], [1e4, 0, 0], [1e4, 0, 0]])```

    - A tensor with a small loss: ```torch.tensor([[1e4, 0, 0], [0, 1e4, 0], [0, 0, 1e4]])```
    """

    def forward(self, logits):
        """"""
        logits = torch.mean(logits, dim=0, keepdim=True)
        return -torch.mean(entropy(logits))
