import torch
from pytorch_metric_learning.utils import common_functions as pml_cf

from ..utils import common_functions as c_f


class GradientReversal(torch.nn.Module):
    """
    Implementation of the gradient reversal layer described in
    [Domain-Adversarial Training of Neural Networks](https://arxiv.org/abs/1505.07818),
    which 'leaves the input unchanged during forward propagation
    and reverses the gradient by multiplying it
    by a negative scalar during backpropagation.'
    """

    def __init__(self, weight: float = 1.0):
        """
        Arguments:
            weight: The gradients  will be multiplied by ```-weight```
                during the backward pass.
        """
        super().__init__()
        self.register_buffer("weight", torch.tensor([weight]))
        pml_cf.add_to_recordable_attributes(self, "weight")

    def update_weight(self, new_weight):
        self.weight[0] = new_weight

    def forward(self, x):
        """"""
        return _GradientReversal.apply(x, pml_cf.to_device(self.weight, x))

    def extra_repr(self):
        """"""
        return c_f.extra_repr(self, ["weight"])


class _GradientReversal(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, weight):
        ctx.weight = weight
        return input

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.weight * grad_output, None
