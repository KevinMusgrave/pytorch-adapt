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
            other_args={"domain": domain},
            logger=self.logger,
        )


class DomainSpecificLogitsHook(LogitsHook, DomainSpecificFeaturesHook):
    pass


class AdaBNHook(BaseWrapperHook):
    """
    Passes inputs into model without doing any optimization.
    The model is expected to receive a ```domain``` argument
    and update its BatchNorm parameters itself.
    """

    def __init__(self, domains=None, **kwargs):
        super().__init__(**kwargs)
        domains = c_f.default(domains, ["src", "target"])
        hooks = []
        for d in domains:
            f_hook = DomainSpecificFeaturesHook(domains=[d], detach=True)
            l_hook = DomainSpecificLogitsHook(domains=[d], detach=True)
            hooks.append(FeaturesChainHook(f_hook, l_hook))
        self.hook = ParallelHook(*hooks)

    def call(self, inputs, losses):
        with torch.no_grad():
            return self.hook(inputs, losses)
