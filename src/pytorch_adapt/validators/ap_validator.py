from torchmetrics.functional import average_precision

from .torchmetrics_validator import TorchmetricsValidator


class APValidator(TorchmetricsValidator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.layer != "preds":
            raise ValueError("layer must be 'preds'")
        if "num_classes" not in self.torchmetric_kwargs:
            raise ValueError("'num_classes' must be provided in torchmetric_kwargs")

    @property
    def accuracy_fn(self):
        return average_precision
