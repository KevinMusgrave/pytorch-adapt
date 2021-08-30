import torch

from ..utils import common_functions as c_f
from .base import BaseWrapperHook
from .features import FeaturesChainHook, FeaturesHook, LogitsHook
from .utils import ParallelHook


class DomainSpecificFeaturesHook(FeaturesHook):
    def add_if_new(
        self, outputs, full_key, output_vals, inputs, model_name, in_keys, domain
    ):
        [domain] = c_f.extract(inputs, [f"{domain}_domain"])
        c_f.add_if_new(
            outputs,
            full_key,
            output_vals,
            inputs,
            model_name,
            in_keys,
            other_args=[domain],
        )


class DomainSpecificLogitsHook(LogitsHook, DomainSpecificFeaturesHook):
    pass


class AdaBNHook(BaseWrapperHook):
    def __init__(self, domains=None, **kwargs):
        super().__init__(**kwargs)
        domains = c_f.default(domains, ["src", "target"])
        hooks = []
        for d in domains:
            f_hook = DomainSpecificFeaturesHook(domains=[d], detach=True)
            l_hook = DomainSpecificLogitsHook(domains=[d], detach=True)
            hooks.append(FeaturesChainHook(f_hook, l_hook))
        self.hook = ParallelHook(*hooks)

    def call(self, losses, inputs):
        with torch.no_grad():
            losses, outputs = self.hook(losses, inputs)
        return losses, outputs
