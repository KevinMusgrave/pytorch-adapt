import torch

from ..layers import UniformDistributionLoss
from .gan import GANHook


class DomainConfusionHook(GANHook):
    def __init__(self, **kwargs):
        super().__init__(
            disc_domain_loss_fn=torch.nn.CrossEntropyLoss(reduction="none"),
            gen_domain_loss_fn=UniformDistributionLoss(),
            **kwargs,
        )
