import torch

from ..containers import KeyEnforcer
from ..hooks import (
    CDANEHook,
    CDANHook,
    DomainConfusionHook,
    GANEHook,
    GANHook,
    VADAHook,
)
from ..utils import common_functions as c_f
from .base_adapter import BaseGCDAdapter


class GAN(BaseGCDAdapter):
    """
    Wraps [GANHook][pytorch_adapt.hooks.gan].
    """

    hook_cls = GANHook

    def init_hook(self, hook_kwargs):
        g_opts = c_f.extract(self.optimizers, ["G", "C"])
        d_opts = [self.optimizers["D"]]
        self.hook = self.hook_cls(d_opts=d_opts, g_opts=g_opts, **hook_kwargs)


class GANE(GAN):
    hook_cls = GANEHook


class CDAN(GAN):
    """
    Wraps [CDANHook][pytorch_adapt.hooks.cdan].
    """

    hook_cls = CDANHook

    def get_key_enforcer(self):
        ke = super().get_key_enforcer()
        ke.requirements["misc"] = ["feature_combiner"]
        return ke


class CDANE(CDAN):
    hook_cls = CDANEHook


class DomainConfusion(GAN):
    """
    Wraps [DomainConfusionHook][pytorch_adapt.hooks.domain_confusion].
    """

    hook_cls = DomainConfusionHook


class VADA(GAN):
    """
    Wraps [VADAHook][pytorch_adapt.hooks.vada].
    """

    hook_cls = VADAHook

    def init_containers_and_check_keys(self):
        models = self.containers["models"]
        misc = self.containers["misc"]
        misc["combined_model"] = torch.nn.Sequential(models["G"], models["C"])
        super().init_containers_and_check_keys()

    def get_key_enforcer(self):
        ke = super().get_key_enforcer()
        ke.requirements["misc"] = ["combined_model"]
        return ke
