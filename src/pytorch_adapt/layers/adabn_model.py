import torch

from ..utils import common_functions as c_f
from .adaptive_batch_norm import (
    AdaptiveBatchNorm2d,
    PopulationBatchNorm2d,
    convert_bn_to_adabn,
    finalize_bn,
    set_curr_domain,
)


class AdaBNModel(torch.nn.Module):
    def __init__(self, model, affine_domain=0, bn_type=None):
        super().__init__()
        bn_type = c_f.default(bn_type, torch.nn.BatchNorm2d)
        convert_bn_to_adabn(model, affine_domain=affine_domain, bn_type=bn_type)
        self.model = model

    def forward(self, x, domain):
        domain = c_f.check_domain(self, domain)
        set_curr_domain(self.model, domain, AdaptiveBatchNorm2d)
        return self.model(x)

    def eval(self):
        super().eval()
        finalize_bn(self, PopulationBatchNorm2d)
