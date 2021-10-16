from ..layers import (
    AdaptiveFeatureNorm,
    BatchSpectralLoss,
    BNMLoss,
    DiversityLoss,
    EntropyLoss,
    MCCLoss,
)
from ..utils import common_functions as c_f
from .base import BaseHook
from .features import FeaturesAndLogitsHook, FeaturesHook


class BaseLossHook(BaseHook):
    def __init__(self, loss_fn, loss_name, layer, f_hook=None, domains=None, **kwargs):
        super().__init__(**kwargs)
        self.loss_fn = loss_fn
        self.loss_name = loss_name
        self.layer = layer
        self.domains = c_f.default(domains, ["src", "target"])
        self.f_hook = c_f.default(
            f_hook, FeaturesAndLogitsHook, {"domains": self.domains}
        )
        self.regex = [f"^{k}" for k in self.domains]

    def call(self, losses, inputs):
        outputs = self.f_hook(losses, inputs)[1]
        strs = c_f.filter(self.f_hook.out_keys, f"_{self.layer}$", self.regex)
        features = c_f.extract([outputs, inputs], strs)
        loss = sum(self.loss_fn(f) for f in features)
        return {self.loss_name: loss}, outputs

    def _loss_keys(self):
        return [self.loss_name]

    def _out_keys(self):
        return self.f_hook.out_keys

    def extra_repr(self):
        return c_f.extra_repr(self, ["layer", "domains"])


class BSPHook(BaseLossHook):
    def __init__(self, loss_fn=None, domains=None, **kwargs):
        loss_fn = c_f.default(loss_fn, BatchSpectralLoss, {"k": 1})
        domains = c_f.default(domains, ["src", "target"])
        f_hook = FeaturesHook(domains=domains)
        super().__init__(
            loss_fn=loss_fn,
            loss_name="bsp_loss",
            domains=domains,
            layer="features",
            f_hook=f_hook,
            **kwargs,
        )


class BNMHook(BaseLossHook):
    def __init__(self, loss_fn=None, domains=None, **kwargs):
        loss_fn = c_f.default(loss_fn, BNMLoss, {})
        domains = c_f.default(domains, ["target"])
        f_hook = FeaturesAndLogitsHook(domains=domains)
        super().__init__(
            loss_fn=loss_fn,
            loss_name="bnm_loss",
            domains=domains,
            layer="logits",
            f_hook=f_hook,
            **kwargs,
        )


class TargetEntropyHook(BaseLossHook):
    def __init__(self, loss_fn=None, domains=None, **kwargs):
        loss_fn = c_f.default(loss_fn, EntropyLoss, {})
        domains = c_f.default(domains, ["target"])
        f_hook = FeaturesAndLogitsHook(domains=domains)
        super().__init__(
            loss_fn=loss_fn,
            loss_name="entropy_loss",
            domains=domains,
            layer="logits",
            f_hook=f_hook,
            **kwargs,
        )


class TargetDiversityHook(BaseLossHook):
    def __init__(self, loss_fn=None, domains=None, **kwargs):
        loss_fn = c_f.default(loss_fn, DiversityLoss, {})
        domains = c_f.default(domains, ["target"])
        f_hook = FeaturesAndLogitsHook(domains=domains)
        super().__init__(
            loss_fn=loss_fn,
            loss_name="diversity_loss",
            domains=domains,
            layer="logits",
            f_hook=f_hook,
            **kwargs,
        )


class MCCHook(BaseLossHook):
    def __init__(self, loss_fn=None, domains=None, **kwargs):
        loss_fn = c_f.default(loss_fn, MCCLoss, {})
        domains = c_f.default(domains, ["target"])
        f_hook = FeaturesAndLogitsHook(domains=domains)
        super().__init__(
            loss_fn=loss_fn,
            loss_name="mcc_loss",
            domains=domains,
            layer="logits",
            f_hook=f_hook,
            **kwargs,
        )


class AFNHook(BaseLossHook):
    def __init__(self, loss_fn=None, domains=None, **kwargs):
        loss_fn = c_f.default(loss_fn, AdaptiveFeatureNorm, {})
        domains = c_f.default(domains, ["src", "target"])
        f_hook = FeaturesHook(domains=domains)
        super().__init__(
            loss_fn=loss_fn,
            loss_name="afn_loss",
            domains=domains,
            layer="features",
            f_hook=f_hook,
            **kwargs,
        )
