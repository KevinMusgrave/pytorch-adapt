import torch

from ..utils import common_functions as c_f
from .base import BaseWrapperHook
from .classification import SoftmaxLocallyHook
from .domain import DomainLossHook
from .features import CombinedFeaturesHook, FeaturesAndLogitsHook
from .gan import GANHook
from .utils import ChainHook


def cdan_key_map(fc_hook):
    strs = c_f.filter(fc_hook.out_keys, "", ["^src", "^target"])
    return {k: v for k, v in zip(strs, ["src_imgs_features", "target_imgs_features"])}


class CDANDomainHook(BaseWrapperHook):
    def __init__(
        self, loss_prefix, detach_features, reverse_labels, softmax=True, **kwargs
    ):
        super().__init__(**kwargs)
        f_hook = FeaturesAndLogitsHook()
        fc_hook = CombinedFeaturesHook()
        key_map = cdan_key_map(fc_hook)
        if softmax:
            strs = c_f.filter(f_hook.out_keys, "_features_logits$", ["^src", "^target"])
            fc_hook = SoftmaxLocallyHook(strs, fc_hook)
        d_hook = DomainLossHook(
            loss_prefix=loss_prefix,
            detach_features=detach_features,
            reverse_labels=reverse_labels,
            key_map=key_map,
        )
        self.hook = ChainHook(f_hook, fc_hook, d_hook)


class CDANDomainHookD(CDANDomainHook):
    def __init__(self, **kwargs):
        super().__init__(
            loss_prefix="d_", detach_features=True, reverse_labels=False, **kwargs
        )


class CDANDomainHookG(CDANDomainHook):
    def __init__(self, **kwargs):
        super().__init__(
            loss_prefix="g_", detach_features=False, reverse_labels=True, **kwargs
        )


class CDANHook(GANHook):
    def __init__(self, softmax=True, **kwargs):
        super().__init__(
            disc_hook=CDANDomainHookD(softmax=softmax),
            gen_hook=CDANDomainHookG(softmax=softmax),
            **kwargs
        )
