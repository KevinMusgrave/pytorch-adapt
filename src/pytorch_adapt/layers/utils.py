import numpy as np
import torch
from pytorch_metric_learning.utils import common_functions as pml_cf

from ..utils import common_functions as c_f


def split_half(x, dim):
    d = x.shape[dim] // 2
    return torch.split(x, d, dim=dim)


def num_elements_minus_diag(x):
    n = x.shape[0]
    return n * (n - 1)


def get_kernel_scales(low=-8, high=8, num_kernels=33, base=2.0):
    return torch.from_numpy(np.logspace(low, high, num=num_kernels, base=base))


def _mmd_dist_mats(x, y, dist_func, bandwidth=None):
    xx = dist_func(x, x)
    yy = dist_func(y, y)
    zz = dist_func(x, y)

    with torch.no_grad():
        # https://arxiv.org/pdf/1409.6041.pdf
        # https://arxiv.org/pdf/1707.07269.pdf
        denom = (
            torch.median(xx)
            if bandwidth is None
            else torch.tensor([bandwidth], dtype=xx.dtype, device=xx.device)
        )
        scale = -1.0 / denom

    return xx, yy, zz, scale


def get_mmd_dist_mats(x, y, dist_func, bandwidth):
    if c_f.is_list_or_tuple(x):
        xx, yy, zz, scale = [], [], [], []
        for i in range(len(x)):
            _xx, _yy, _zz, _scale = _mmd_dist_mats(x[i], y[i], dist_func, bandwidth)
            xx.append(_xx)
            yy.append(_yy)
            zz.append(_zz)
            scale.append(_scale)
        return xx, yy, zz, scale
    else:
        return _mmd_dist_mats(x, y, dist_func, bandwidth)


def get_default_kernel_weights(scale):
    if torch.is_tensor(scale) and torch.numel(scale) > 1:
        return torch.ones_like(scale) / len(scale)
    else:
        return 1


def _mmd_quadratic(x, scale, weights):
    return torch.sum(torch.exp(x.unsqueeze(2) * scale) * weights, dim=2)


def get_mmd_quadratic(xx, yy, zz, scale, weights=None):
    # https://jmlr.csail.mit.edu/papers/volume13/gretton12a/gretton12a.pdf
    # https://arxiv.org/pdf/1502.02791.pdf
    is_joint_mmd = c_f.is_list_or_tuple(xx)

    if is_joint_mmd:
        xx_prod, yy_prod, zz_prod = 1, 1, 1
        for i in range(len(xx)):
            curr_weights = c_f.default(weights, get_default_kernel_weights(scale[i]))
            xx_prod *= _mmd_quadratic(xx[i], scale[i], curr_weights)
            yy_prod *= _mmd_quadratic(yy[i], scale[i], curr_weights)
            zz_prod *= _mmd_quadratic(zz[i], scale[i], curr_weights)
        xx_prod.fill_diagonal_(0)
        yy_prod.fill_diagonal_(0)
        xx, yy, zz = xx_prod, yy_prod, zz_prod
    else:
        weights = c_f.default(weights, get_default_kernel_weights(scale))
        xx = _mmd_quadratic(xx, scale, weights).fill_diagonal_(0)
        yy = _mmd_quadratic(yy, scale, weights).fill_diagonal_(0)
        zz = _mmd_quadratic(zz, scale, weights)

    xx_scaler = 1.0 / num_elements_minus_diag(xx)
    yy_scaler = 1.0 / num_elements_minus_diag(yy)
    return xx_scaler * torch.sum(xx) + yy_scaler * torch.sum(yy) - 2 * torch.mean(zz)


def _mmd_linear(x, i, j, scale, weights):
    return torch.sum(torch.exp(x[i, j] * scale) * weights, dim=0)


def _mmd_linear_helper(xx, yy, zz, scale, weights):
    B = xx.shape[0]
    idx_range = torch.arange(0, B // 2, device=xx.device)
    s1 = idx_range * 2
    s2 = s1 + 1

    if scale.ndim == 0:
        scale = scale.unsqueeze(0)
    scale = scale.unsqueeze(1)
    weights = c_f.default(weights, get_default_kernel_weights(scale))

    loss1 = _mmd_linear(xx, s1, s2, scale, weights)
    loss2 = _mmd_linear(yy, s1, s2, scale, weights)
    loss3 = _mmd_linear(zz, s1, s2, scale, weights)
    loss4 = _mmd_linear(zz, s2, s1, scale, weights)

    return loss1, loss2, loss3, loss4


def get_mmd_linear(xx, yy, zz, scale, weights=None):
    # https://jmlr.csail.mit.edu/papers/volume13/gretton12a/gretton12a.pdf
    # https://arxiv.org/pdf/1502.02791.pdf
    is_joint_mmd = c_f.is_list_or_tuple(xx)
    B = xx[0].shape[0] if is_joint_mmd else xx.shape[0]

    if is_joint_mmd:
        product_list = [1, 1, 1, 1]
        for i in range(len(xx)):
            curr_kernels = _mmd_linear_helper(xx[i], yy[i], zz[i], scale[i], weights)
            product_list = [a * b for a, b in zip(product_list, curr_kernels)]
        loss1, loss2, loss3, loss4 = [torch.sum(a) for a in product_list]
    else:
        loss1, loss2, loss3, loss4 = [
            torch.sum(a) for a in _mmd_linear_helper(xx, yy, zz, scale, weights)
        ]

    loss = loss1 + loss2 - loss3 - loss4
    return torch.sum(loss) / float(B // 2)


def _mmd_quadratic_batched(rsum, scale, weights, query_is_ref):
    def fn(mat, s, *_):
        if query_is_ref:
            mat = c_f.mask_out_self(mat, s)
        rsum[0] += torch.sum(_mmd_quadratic(mat, scale, weights))

    return fn


def get_median_of_medians(x, dist_func):
    medians = []

    def fn(mat, *_):
        with torch.no_grad():
            medians.append(torch.median(mat))

    dist_func.iter_fn = fn
    dist_func(x, x)
    return torch.median(torch.stack(medians))


def get_mmd_quadratic_batched(x, y, dist_func, kernel_scales, bandwidth, weights=None):
    if torch.is_tensor(kernel_scales):
        kernel_scales = pml_cf.to_device(kernel_scales, x, dtype=x.dtype)
    if bandwidth is None:
        bandwidth = get_median_of_medians(x, dist_func)
    scale = -kernel_scales / bandwidth
    weights = c_f.default(weights, get_default_kernel_weights(scale))

    sums = []
    for s, t in [(x, x), (y, y), (x, y)]:
        rsum = [0]
        query_is_ref = s is t
        dist_func.iter_fn = _mmd_quadratic_batched(rsum, scale, weights, query_is_ref)
        dist_func(s, t)
        denom = (len(s) * (len(s) - 1)) if query_is_ref else (len(s) * len(t))
        sums.append(torch.sum(rsum[0]) / denom)

    return sums[0] + sums[1] - 2 * sums[2]
