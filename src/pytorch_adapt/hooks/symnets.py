from ..layers import (
    SymNetsCategoryLossListInput,
    SymNetsDomainLoss,
    SymNetsEntropyLossListInput,
)
from ..utils import common_functions as c_f
from .base import BaseWrapperHook
from .classification import CLossHook
from .features import FeaturesAndLogitsHook, FeaturesHook
from .losses import TargetEntropyHook
from .mcd import MultipleCLossHook
from .optimizer import OptimizerHook, SummaryHook
from .utils import ChainHook


class SymNetsDomainLossHook(BaseWrapperHook):
    # domain == src or target
    def __init__(self, domain, half_idx, detach_features=False, **kwargs):
        super().__init__(**kwargs)
        self.domain = domain
        self.half_idx = half_idx
        self.hook = FeaturesAndLogitsHook(
            domains=[domain], detach_features=detach_features
        )
        self.loss_fn = SymNetsDomainLoss(half_idx=half_idx)

    def call(self, losses, inputs):
        outputs = self.hook(losses, inputs)[1]
        [logits] = c_f.extract(
            [outputs, inputs], c_f.filter(self.hook.out_keys, f"_logits$")
        )
        loss = self.loss_fn(*logits)
        return {self._loss_keys()[0]: loss}, outputs

    def _loss_keys(self):
        return [f"symnets_{self.domain}_domain_loss_{self.half_idx}"]


class SymNetsCategoryLossHook(CLossHook):
    def __init__(self, **kwargs):
        super().__init__(loss_fn=SymNetsCategoryLossListInput(), **kwargs)

    def _loss_keys(self):
        return ["symnets_category_loss"]


class SymNetsEntropyHook(TargetEntropyHook):
    def __init__(self, **kwargs):
        super().__init__(
            loss_fn=SymNetsEntropyLossListInput(), loss_prefix="symnets_", **kwargs
        )


class SymNetsCHook(BaseWrapperHook):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        c_loss_hook = MultipleCLossHook(detach_features=True)
        d_loss_hook1 = SymNetsDomainLossHook(
            "src", 0, detach_features=True, loss_prefix="c_"
        )
        d_loss_hook2 = SymNetsDomainLossHook(
            "target", 1, detach_features=True, loss_prefix="c_"
        )
        self.hook = ChainHook(c_loss_hook, d_loss_hook1, d_loss_hook2)


class SymNetsGHook(BaseWrapperHook):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        category_loss_hook = SymNetsCategoryLossHook()
        d_loss_hook1 = SymNetsDomainLossHook("target", 0, loss_prefix="g_")
        d_loss_hook2 = SymNetsDomainLossHook("target", 1, loss_prefix="g_")
        entropy_hook = SymNetsEntropyHook()
        self.hook = ChainHook(
            category_loss_hook, d_loss_hook1, d_loss_hook2, entropy_hook
        )


class SymNetsHook(BaseWrapperHook):
    """
    Implementation of
    [Domain-Symmetric Networks for Adversarial Domain Adaptation](https://arxiv.org/abs/1904.04663).
    """

    def __init__(
        self,
        c_opts,
        g_opts,
        c_weighter=None,
        c_reducer=None,
        g_weighter=None,
        g_reducer=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        f_hook = FeaturesHook()
        c_hook = OptimizerHook(SymNetsCHook(), c_opts, c_weighter, c_reducer)
        g_hook = OptimizerHook(SymNetsGHook(), g_opts, g_weighter, g_reducer)
        s_hook = SummaryHook({"c_loss": c_hook, "g_loss": g_hook})
        self.hook = ChainHook(f_hook, c_hook, g_hook, s_hook)
