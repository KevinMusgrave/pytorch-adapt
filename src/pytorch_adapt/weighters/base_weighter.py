from typing import Callable, Dict, List, Tuple

import torch
from pytorch_metric_learning.utils import common_functions as pml_cf

from ..utils import common_functions as c_f


def weight_losses(reduction, weights, scale, loss_dict):
    if "total" in loss_dict:
        raise ValueError("'total' is a reserved key. Use a different loss name")
    c_f.assert_keys_are_present(weights, "weights", loss_dict)
    losses = []
    components = {}
    for k, v in loss_dict.items():
        try:
            loss = v if k not in weights else v * weights[k]
            loss = loss * scale
            losses.append(loss)
            components[k] = loss.item()
        except Exception as e:
            c_f.append_error_message(e, f"\nError occuring with loss key = {k}")
            raise

    total = reduction(losses)
    components["total"] = total.item()
    return total, components


class BaseWeighter:
    """
    Multiplies losses by scalar values, and then reduces them to a single value.
    """

    def __init__(
        self,
        reduction: Callable[[List[torch.Tensor]], torch.Tensor],
        weights: Dict[str, float] = None,
        scale: float = 1,
    ):
        """
        Arguments:
            reduction: A function that takes in a list of losses and returns a single loss value.
            weights: A mapping from loss names to weight values. If ```None```, weights are assumed to be 1.
            scale: A scalar that every loss gets multiplied by.
        """
        self.reduction = reduction
        self.weights = c_f.default(weights, {})
        self.scale = scale
        pml_cf.add_to_recordable_attributes(self, list_of_names=["weights", "scale"])

    def __call__(
        self, loss_dict: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Arguments:
            loss_dict: A mapping from loss names to loss values.
        Returns:
            A tuple where ```tuple[0]``` is the loss that ```.backward()``` can be called on,
            and ```tuple[1]``` is a dictionary of floats (detached from the autograd graph)
            that contains the weighted loss components.
        """
        return weight_losses(self.reduction, self.weights, self.scale, loss_dict)

    def __repr__(self):
        return c_f.nice_repr(self, c_f.extra_repr(self, ["weights", "scale"]), {})
