import torch

from ..utils import common_functions as c_f


class ConcatSoftmax(torch.nn.Module):
    """
    Applies softmax to the concatenation of a list of tensors.
    """

    def __init__(self, dim: int = 1):
        """
        Arguments:
            dim: a dimension along which softmax will be computed
        """
        super().__init__()
        self.dim = dim

    def forward(self, *x: torch.Tensor):
        """
        Arguments:
            *x: A sequence of tensors to be concatenated
        """
        all_logits = torch.cat(x, dim=self.dim)
        return torch.nn.functional.softmax(all_logits, dim=self.dim)

    def extra_repr(self):
        """"""
        return c_f.extra_repr(self, ["dim"])
