import numpy as np
import torch
from pytorch_metric_learning.distances import LpDistance
from pytorch_metric_learning.utils import common_functions as pml_cf

from ..utils import common_functions as c_f
from . import utils as l_u


class MMDLoss(torch.nn.Module):
    def __init__(self, kernel_scales=1, mmd_type="linear"):
        super().__init__()
        self.kernel_scales = kernel_scales
        self.dist_func = LpDistance(normalize_embeddings=False, p=2, power=2)
        self.mmd_type = mmd_type
        if mmd_type == "linear":
            self.mmd_func = l_u.get_mmd_linear
        elif mmd_type == "quadratic":
            self.mmd_func = l_u.get_mmd_quadratic
        else:
            raise ValueError("mmd_type must be either linear or quadratic")

    # input can be embeddings or list of embeddings
    def forward(self, x, y):
        xx, yy, zz, scale = l_u.get_mmd_dist_mats(x, y, self.dist_func)
        if torch.is_tensor(self.kernel_scales):
            s = scale[0] if c_f.is_list_or_tuple(scale) else scale
            self.kernel_scales = pml_cf.to_device(self.kernel_scales, s, dtype=s.dtype)

        if c_f.is_list_or_tuple(scale):
            for i in range(len(scale)):
                scale[i] = scale[i] * self.kernel_scales
        else:
            scale = scale * self.kernel_scales

        return self.mmd_func(xx, yy, zz, scale)

    def extra_repr(self):
        return c_f.extra_repr(self, ["mmd_type", "kernel_scales"])
