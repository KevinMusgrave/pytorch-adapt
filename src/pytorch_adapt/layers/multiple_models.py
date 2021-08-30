import copy

import torch


class MultipleModels(torch.nn.Module):
    def __init__(self, *models):
        super().__init__()
        self.models = torch.nn.ModuleList(models)

    def forward(self, x):
        outputs = []
        for m in self.models:
            outputs.append(m(x))
        return outputs
