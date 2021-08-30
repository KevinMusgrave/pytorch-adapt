import torch


class PlusResidual(torch.nn.Module):
    def __init__(self, residual):
        super().__init__()
        self.residual = residual

    def forward(self, x):
        return x + self.residual(x)
