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
    def __init__(self, reduction, weights=None, scale=1):
        self.reduction = reduction
        self.weights = c_f.default(weights, {})
        self.scale = scale
        pml_cf.add_to_recordable_attributes(self, list_of_names=["weights", "scale"])

    def __call__(self, loss_dict):
        return weight_losses(self.reduction, self.weights, self.scale, loss_dict)

    def __repr__(self):
        return c_f.nice_repr(self, c_f.extra_repr(self, ["weights", "scale"]), {})
