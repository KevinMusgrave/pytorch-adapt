import torch
from pytorch_metric_learning.utils import common_functions as pml_cf


class NLLLoss(torch.nn.Module):
    """
    Same as torch.nn.NLLLoss but takes in softmax as input
    """

    def forward(self, x: torch.Tensor, y: torch.Tensor):
        """ """
        x = torch.log(x + pml_cf.small_val(x.dtype))
        return torch.nn.functional.nll_loss(x, y)
