import torch

from ..layers import ISTLoss
from ..utils import common_functions as c_f
from .base import BaseWrapperHook
from .features import FeaturesHook


class ISTLossHook(BaseWrapperHook):
    def __init__(self, distance=None, with_div=True, **kwargs):
        super().__init__(**kwargs)
        self.loss_fn = ISTLoss(distance=distance, with_div=with_div)
        self.hook = FeaturesHook()

    def call(self, losses, inputs):
        outputs = self.hook(losses, inputs)[1]
        [src_features, target_features] = c_f.extract(
            [outputs, inputs],
            c_f.filter(self.hook.out_keys, "_features$", ["^src", "^target"]),
        )
        features = torch.cat([src_features, target_features], dim=0)
        [src_domain, target_domain] = c_f.extract(
            inputs, ["src_domain", "target_domain"]
        )
        domain = torch.cat([src_domain, target_domain], dim=0)
        loss = self.loss_fn(features, domain)
        return {"ist_loss": loss}, outputs

    def _loss_keys(self):
        return ["ist_loss"]
