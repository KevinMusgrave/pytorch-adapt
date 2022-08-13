from torchmetrics.functional import average_precision

from .torchmetrics_validator import TorchmetricsValidator


class APValidator(TorchmetricsValidator):
    @property
    def accuracy_fn(self):
        return average_precision
