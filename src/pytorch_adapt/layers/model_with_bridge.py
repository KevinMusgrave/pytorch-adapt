import copy

import torch

from ..utils import common_functions as c_f


class ModelWithBridge(torch.nn.Module):
    def __init__(self, model, bridge=None):
        super().__init__()
        self.model = model
        if bridge is None:
            bridge = c_f.reinit(copy.deepcopy(model))
        self.bridge = bridge

    def forward(self, x, return_bridge=False):
        y = self.model(x)
        z = self.bridge(x)
        output = y - z
        if return_bridge:
            return output, z
        return output
