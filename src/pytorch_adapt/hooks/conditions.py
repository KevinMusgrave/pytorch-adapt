import torch

from ..layers import SufficientAccuracy
from ..utils import common_functions as c_f
from .base import BaseConditionHook
from .features import DLogitsHook, FeaturesChainHook, FeaturesHook


class StrongDHook(BaseConditionHook):
    def __init__(self, threshold=0.6, **kwargs):
        super().__init__(**kwargs)
        self.accuracy_fn = SufficientAccuracy(
            threshold=threshold, to_probs_func=torch.nn.Sigmoid()
        )
        self.hook = FeaturesChainHook(
            FeaturesHook(detach=True), DLogitsHook(detach=True)
        )

    def call(self, losses, inputs):
        with torch.no_grad():
            outputs = self.hook(losses, inputs)[1]
            [d_src_logits, d_target_logits] = c_f.extract(
                [outputs, inputs],
                c_f.filter(
                    self.hook.out_keys, "_dlogits_detached$", ["^src", "^target"]
                ),
            )
            [src_domain, target_domain] = c_f.extract(
                inputs, ["src_domain", "target_domain"]
            )
            dlogits = torch.cat([d_src_logits, d_target_logits], dim=0)
            domain_labels = torch.cat([src_domain, target_domain], dim=0)
            return self.accuracy_fn(dlogits, domain_labels)
