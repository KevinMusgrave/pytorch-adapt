from typing import Any, List

import torch


class MultipleModels(torch.nn.Module):
    """
    Wraps a list of models, and returns their outputs as a list of tensors
    """

    def __init__(self, *models: torch.nn.Module):
        """
        Arguments:
            models: The models to be wrapped.
        """
        super().__init__()
        self.models = torch.nn.ModuleList(models)

    def forward(self, x: torch.Tensor) -> List[Any]:
        """
        Arguments:
            x: the input to each model
        Returns:
            A list containing the output of each model.
        """
        outputs = []
        for m in self.models:
            outputs.append(m(x))
        return outputs
