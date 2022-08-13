from torchmetrics.functional import accuracy

from .torchmetrics_validator import TorchmetricsValidator


class AccuracyValidator(TorchmetricsValidator):
    """
    Returns accuracy using the
    [torchmetrics accuracy function](https://torchmetrics.readthedocs.io/en/latest/references/functional.html#accuracy-func).

    The required dataset splits are ```["src_val"]```.
    This can be changed using [```key_map```][pytorch_adapt.validators.BaseValidator.__init__].
    """

    @property
    def accuracy_fn(self):
        return accuracy
