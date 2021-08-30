from .base_weighter import BaseWeighter
from .mean_weighter import MeanWeighter, mean_weighter
from .sum_weighter import SumWeighter


def total_loss(loss_components):
    return loss_components["total"]


def only_components(loss_components):
    return {k: v for k, v in loss_components.items() if k != "total"}


def get_multiple_loss_totals(multiple_loss_components):
    return {k: total_loss(v) for k, v in multiple_loss_components.items()}
