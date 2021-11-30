from collections.abc import MutableMapping

import torch


def default_optimizer_tuple():
    return (torch.optim.Adam, {"lr": 0.0001})


def with_opt(x):
    suffix = "_opt"
    if isinstance(x, str):
        return f"{x}{suffix}"
    if isinstance(x, list):
        return [with_opt(y) for y in x]
    if isinstance(x, MutableMapping):
        return {with_opt(k): v for k, v in x.items()}
