import torch
from pytorch_metric_learning.utils import common_functions as pml_cf

from ..utils import common_functions as c_f


class GradientReversal(torch.nn.Module):
    def __init__(self, weight=1.0):
        super().__init__()
        self.register_buffer("weight", torch.tensor([weight]))
        pml_cf.add_to_recordable_attributes(self, "weight")

    def update_weight(self, new_weight):
        self.weight[0] = new_weight

    def forward(self, x):
        return _GradientReversal.apply(x, pml_cf.to_device(self.weight, x))

    def extra_repr(self):
        return c_f.extra_repr(self, ["weight"])


class _GradientReversal(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, weight):
        ctx.weight = weight
        return input

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.weight * grad_output, None
