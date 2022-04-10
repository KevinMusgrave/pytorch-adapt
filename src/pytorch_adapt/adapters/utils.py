from collections.abc import MutableMapping
from typing import Any, Dict, Tuple

import torch


def default_optimizer_tuple() -> Tuple[torch.optim.Optimizer, Dict[str, Any]]:
    """
    Returns:
        A tuple to be passed into an [Optimizers][pytorch_adapt.containers.Optimizers]
        container. The tuple specifies an Adam optimizer with lr 0.0001.
    """
    return (torch.optim.Adam, {"lr": 0.0001})


def with_opt(x):
    suffix = "_opt"
    if isinstance(x, str):
        return f"{x}{suffix}"
    if isinstance(x, list):
        return [with_opt(y) for y in x]
    if isinstance(x, MutableMapping):
        return {with_opt(k): v for k, v in x.items()}


def container_names():
    return ["models", "optimizers", "lr_schedulers", "misc"]
