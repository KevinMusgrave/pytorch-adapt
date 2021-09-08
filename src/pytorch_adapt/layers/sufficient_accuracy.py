from typing import Callable

import torch
from pytorch_metric_learning.utils import common_functions as pml_cf
from torchmetrics.functional import accuracy

from ..utils import common_functions as c_f


class SufficientAccuracy(torch.nn.Module):
    """
    Determines if a batch of logits has accuracy greater
    than some threshold. This can be used to control
    program flow.

    Example:
    ```python
    condition_fn = SufficientAccuracy(threshold=0.7)
    if condition_fn(logits, labels):
        ...
    ```
    """

    def __init__(
        self,
        threshold: float,
        accuracy_func: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = None,
        to_probs_func: Callable[[torch.Tensor], torch.Tensor] = None,
    ):
        """
        Arguments:
            threshold: The accuracy must be greater than this
                for the forward pass to return True.
            accuracy_func: function that takes in ```(to_probs_func(logits), labels)```
                and returns accuracy. If ```None```, then classification accuracy is used.
            to_probs_func: function that processes the logits before they get passed
                to ```accuracy_func```. If ```None```, then ```torch.nn.Sigmoid``` is used
        """

        super().__init__()
        self.threshold = threshold
        self.accuracy_func = c_f.default(accuracy_func, accuracy)
        self.to_probs_func = c_f.default(to_probs_func, torch.nn.Sigmoid())
        pml_cf.add_to_recordable_attributes(
            self, list_of_names=["accuracy", "threshold"]
        )

    def forward(self, x: torch.Tensor, labels: torch.Tensor) -> bool:
        """
        Arguments:
            x: logits to compute accuracy for
            labels: the corresponding labels
        Returns:
            ```True``` if the accuracy is greater than ```self.threshold```
        """
        with torch.no_grad():
            x = self.to_probs_func(x)
            labels = labels.type(torch.int)
            self.accuracy = self.accuracy_func(x, labels).item()
        return self.accuracy > self.threshold

    def extra_repr(self):
        """"""
        return c_f.extra_repr(self, ["threshold"])
