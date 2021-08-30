import math

import torch
from pytorch_metric_learning.utils import common_functions as pml_cf

from ..utils import common_functions as c_f


# modified from https://github.com/thuml/CDAN/blob/master/pytorch/network.py
class RandomizedDotProduct(torch.nn.Module):
    def __init__(self, in_dims, out_dim=1024):
        super().__init__()
        self.in_dims = in_dims
        for i, d in enumerate(in_dims):
            self.register_buffer(self.rand_mat_name(i), torch.randn(d, out_dim))
        self.out_dim = out_dim
        self.num_mats = len(in_dims)
        self.divisor = math.pow(float(self.out_dim), 1.0 / self.num_mats)

    def forward(self, *inputs):
        for i in range(self.num_mats):
            # move to device if necessary
            curr = inputs[i]
            self.set_rand_mat(
                i, pml_cf.to_device(self.get_rand_mat(i), curr, dtype=curr.dtype)
            )

        return_list = [
            torch.mm(inputs[i], self.get_rand_mat(i)) for i in range(self.num_mats)
        ]
        return_tensor = return_list[0] / self.divisor
        for single in return_list[1:]:
            return_tensor = return_tensor * single
        return return_tensor

    def set_rand_mat(self, i, value):
        setattr(self, self.rand_mat_name(i), value)

    def get_rand_mat(self, i):
        return getattr(self, self.rand_mat_name(i))

    def rand_mat_name(self, i):
        return f"rand_mat{i}"

    def extra_repr(self):
        return c_f.extra_repr(self, ["in_dims", "out_dim", "divisor"])
