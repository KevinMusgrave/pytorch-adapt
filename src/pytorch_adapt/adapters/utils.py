import torch

from ..utils import common_functions as c_f


def default_optimizer_tuple():
    return (torch.optim.Adam, {"lr": 0.0001})
