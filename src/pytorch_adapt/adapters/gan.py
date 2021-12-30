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
from .base_adapter import BaseGCDAdapter
from .utils import with_opt


class GAN(BaseGCDAdapter):
    """
    Extends [BaseGCDAdapter][pytorch_adapt.adapters.base_adapter.BaseGCDAdapter]
    and wraps [GANHook][pytorch_adapt.hooks.GANHook].

    |Container|Required keys|
    |---|---|
    |models|```["G", "C", "D"]```|
    |optimizers|```["G", "C", "D"]```|
    """

    hook_cls = GANHook

    def init_hook(self, hook_kwargs):
        g_opts = with_opt(["G", "C"])
        d_opts = with_opt(["D"])
        self.hook = self.hook_cls(d_opts=d_opts, g_opts=g_opts, **hook_kwargs)


class GANE(GAN):
    hook_cls = GANEHook


class CDAN(GAN):
    """
    Extends [GAN][pytorch_adapt.adapters.GAN]
    and wraps [CDANHook][pytorch_adapt.hooks.CDANHook].

    |Container|Required keys|
    |---|---|
    |models|```["G", "C", "D"]```|
    |optimizers|```["G", "C", "D"]```|
    |misc|```["feature_combiner"]```|
    """

    hook_cls = CDANHook

    def get_key_enforcer(self) -> KeyEnforcer:
        ke = super().get_key_enforcer()
        ke.requirements["misc"] = ["feature_combiner"]
        return ke


class CDANE(CDAN):
    hook_cls = CDANEHook


class DomainConfusion(GAN):
    """
    Extends [GAN][pytorch_adapt.adapters.GAN]
    and wraps [DomainConfusionHook][pytorch_adapt.hooks.DomainConfusionHook].

    |Container|Required keys|
    |---|---|
    |models|```["G", "C", "D"]```|
    |optimizers|```["G", "C", "D"]```|
    """

    hook_cls = DomainConfusionHook


class VADA(GAN):
    """
    Extends [GAN][pytorch_adapt.adapters.GAN]
    and wraps [VADAHook][pytorch_adapt.hooks.VADAHook].

    |Container|Required keys|
    |---|---|
    |models|```["G", "C", "D"]```|
    |optimizers|```["G", "C", "D"]```|
    |misc|```["combined_model"]```|

    The ```"combined_model"``` key does not need to be passed in.
    It is simply ```torch.nn.Sequential(G, C)```, and is set
    automatically during initialization.
    """

    hook_cls = VADAHook

    def init_containers_and_check_keys(self):
        models = self.containers["models"]
        misc = self.containers["misc"]
        misc["combined_model"] = torch.nn.Sequential(models["G"], models["C"])
        super().init_containers_and_check_keys()

    def get_key_enforcer(self) -> KeyEnforcer:
        ke = super().get_key_enforcer()
        ke.requirements["misc"] = ["combined_model"]
        return ke
