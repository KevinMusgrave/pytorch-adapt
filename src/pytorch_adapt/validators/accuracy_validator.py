from torchmetrics.functional import accuracy

from ..utils import common_functions as c_f
from .base_validator import BaseValidator


class AccuracyValidator(BaseValidator):
    """
    Returns accuracy using the
    [torchmetrics accuracy function](https://torchmetrics.readthedocs.io/en/latest/references/functional.html#accuracy-func).

    The required dataset splits are ```["src_val"]```.
    This can be changed using [```key_map```][pytorch_adapt.validators.BaseValidator.__init__].
    """

    def __init__(self, layer="preds", torchmetric_kwargs=None, **kwargs):
        """
        Arguments:
            layer: The name of the layer to pass into the accuracy function.
                For example, ```"preds"``` refers to the softmaxed logits.
            torchmetric_kwargs: A dictionary of keyword arguments to be
                passed into the
                [torchmetrics accuracy function](https://torchmetrics.readthedocs.io/en/latest/references/functional.html#accuracy-func).
        """
        super().__init__(**kwargs)
        self.layer = layer
        self.torchmetric_kwargs = c_f.default(torchmetric_kwargs, {})
        self.accuracy_func = accuracy

    def compute_score(self, src_val):
        return self.accuracy_func(
            src_val[self.layer], src_val["labels"], **self.torchmetric_kwargs
        ).item()
