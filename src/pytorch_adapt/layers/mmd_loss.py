from typing import List, Union

import torch
from pytorch_metric_learning.distances import BatchedDistance, LpDistance
from pytorch_metric_learning.utils import common_functions as pml_cf

from ..utils import common_functions as c_f
from . import utils as l_u


def check_batch_sizes(s, t, mmd_type):
    if mmd_type == "quadratic":
        return
    is_list = c_f.is_list_or_tuple(s)
    if (is_list and any(s[i].shape != t[i].shape for i in range(len(s)))) or (
        not is_list and s.shape != t.shape
    ):
        raise ValueError(
            "For mmd_type 'linear', source and target must have the same batch size."
        )


class MMDLoss(torch.nn.Module):
    """
    Implementation of

    - [Learning Transferable Features with Deep Adaptation Networks](https://arxiv.org/abs/1502.02791)

    - [Deep Transfer Learning with Joint Adaptation Networks](https://arxiv.org/abs/1605.06636).
    """

    def __init__(
        self,
        kernel_scales: Union[float, torch.Tensor] = 1,
        mmd_type: str = "linear",
        dist_func=None,
        bandwidth=None,
    ):
        """
        Arguments:
            kernel_scales: The kernel bandwidth is scaled by this amount.
                If a tensor, then multiple kernel bandwidths are used.
            mmd_type: 'linear' or 'quadratic'. 'linear' uses the linear estimate of MK-MMD.
        """
        super().__init__()
        self.kernel_scales = kernel_scales
        self.dist_func = c_f.default(
            dist_func, LpDistance(normalize_embeddings=False, p=2, power=2)
        )
        self.bandwidth = bandwidth
        self.mmd_type = mmd_type
        if mmd_type == "linear":
            self.mmd_func = l_u.get_mmd_linear
        elif mmd_type == "quadratic":
            self.mmd_func = l_u.get_mmd_quadratic
        else:
            raise ValueError("mmd_type must be either linear or quadratic")

    # input can be embeddings or list of embeddings
    def forward(
        self,
        x: Union[torch.Tensor, List[torch.Tensor]],
        y: Union[torch.Tensor, List[torch.Tensor]],
    ) -> torch.Tensor:
        """
        Arguments:
            x: features or a list of features from one domain.
            y: features or a list of features from the other domain.
        Returns:
            MMD if the inputs are tensors, and Joint MMD (JMMD) if the inputs are lists of tensors.
        """
        check_batch_sizes(x, y, self.mmd_type)
        xx, yy, zz, scale = l_u.get_mmd_dist_mats(x, y, self.dist_func, self.bandwidth)
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
        """"""
        return c_f.extra_repr(self, ["mmd_type", "kernel_scales"])


class MMDBatchedLoss(MMDLoss):
    def __init__(self, batch_size=1024, **kwargs):
        super().__init__(**kwargs)
        if self.mmd_type != "quadratic":
            raise ValueError("mmd_type must be 'quadratic'")
        self.mmd_func = l_u.get_mmd_quadratic_batched
        self.dist_func = BatchedDistance(self.dist_func, batch_size=batch_size)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Arguments:
            x: features from one domain.
            y: features from the other domain.
        Returns:
            MMD
        """
        if c_f.is_list_or_tuple(x) or c_f.is_list_or_tuple(y):
            raise TypeError("List of features not yet supported")
        check_batch_sizes(x, y, self.mmd_type)
        return self.mmd_func(x, y, self.dist_func, self.kernel_scales, self.bandwidth)
