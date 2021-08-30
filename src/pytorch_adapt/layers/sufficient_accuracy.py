import torch
from pytorch_metric_learning.utils import common_functions as pml_cf
from torchmetrics.functional import accuracy

from ..utils import common_functions as c_f


class SufficientAccuracy(torch.nn.Module):
    def __init__(self, threshold, accuracy_func=None, to_probs_func=None):
        super().__init__()
        self.threshold = threshold
        self.accuracy_func = c_f.default(accuracy_func, accuracy)
        self.to_probs_func = c_f.default(to_probs_func, torch.nn.Sigmoid())
        pml_cf.add_to_recordable_attributes(
            self, list_of_names=["accuracy", "threshold"]
        )

    def forward(self, x, labels):
        with torch.no_grad():
            x = self.to_probs_func(x)
            labels = labels.type(torch.int)
            self.accuracy = self.accuracy_func(x, labels).item()
        return self.accuracy > self.threshold

    def extra_repr(self):
        return c_f.extra_repr(self, ["threshold"])
