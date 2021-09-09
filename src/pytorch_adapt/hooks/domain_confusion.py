import torch

from ..layers import UniformDistributionLoss
from .gan import GANHook


class DomainConfusionHook(GANHook):
    """
    Implementation of
    [Simultaneous Deep Transfer Across Domains and Tasks](https://arxiv.org/abs/1510.02192)

    Extends [```GANHook```][pytorch_adapt.hooks.gan.GANHook].
    """

    def __init__(self, **kwargs):
        super().__init__(
            disc_domain_loss_fn=torch.nn.CrossEntropyLoss(reduction="none"),
            gen_domain_loss_fn=UniformDistributionLoss(),
            **kwargs,
        )
