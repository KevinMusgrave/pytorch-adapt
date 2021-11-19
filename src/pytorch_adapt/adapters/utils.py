import torch


def default_optimizer_tuple():
    return (torch.optim.Adam, {"lr": 0.0001})
