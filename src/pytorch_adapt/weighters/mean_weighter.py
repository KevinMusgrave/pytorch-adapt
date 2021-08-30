from .base_weighter import BaseWeighter, weight_losses


def mean(losses):
    return sum(losses) / len(losses)


def mean_weighter(kwargs):
    return weight_losses(mean, {}, 1, kwargs)


class MeanWeighter(BaseWeighter):
    def __init__(self, **kwargs):
        super().__init__(reduction=mean, **kwargs)
