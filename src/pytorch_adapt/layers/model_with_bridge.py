import copy
from typing import Tuple, Union

import torch

from ..utils import common_functions as c_f


class ModelWithBridge(torch.nn.Module):
    """
    Implementation of the bridge architecture described in
    [Gradually Vanishing Bridge for Adversarial Domain Adaptation](https://arxiv.org/abs/2003.13183).
    """

    def __init__(self, model: torch.nn.Module, bridge: torch.nn.Module = None):
        """
        Arguments:
            model: Any pytorch model.
            bridge: A model which has the same input/output sizes as ```model```.
                If ```None```, then the bridge is formed by copying ```model```,
                and randomly reinitialization all its parameters.
        """
        super().__init__()
        self.model = model
        if bridge is None:
            bridge = c_f.reinit(copy.deepcopy(model))
        self.bridge = bridge

    def forward(
        self, x: torch.Tensor, return_bridge: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Arguments:
            x: The input to both ```self.model``` and ```self.bridge```.
            return_bridge: Whether or not to return the bridge output
                in addition to the ```model - bridge``` output
        Returns:
            If ```return_bridge = False```, then return just ```model - bridge```.
            If ```return_bridge = True```, then return a tuple of ```(model - bridge), bridge```
        """
        y = self.model(x)
        z = self.bridge(x)
        output = y - z
        if return_bridge:
            return output, z
        return output
