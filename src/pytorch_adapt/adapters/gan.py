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
    Wraps [GANHook][pytorch_adapt.hooks.GANHook].

    |Container|Required keys|
    |---|---|
    |models|```["G", "C", "D"]```|
    |optimizers|```["G", "C", "D"]```|
    """

    def init_hook(self, hook_kwargs):
        g_opts = with_opt(["G", "C"])
        d_opts = with_opt(["D"])
        self.hook = self.hook_cls(d_opts=d_opts, g_opts=g_opts, **hook_kwargs)

    @property
    def hook_cls(self):
        return GANHook


class GANE(GAN):
    @property
    def hook_cls(self):
        return GANEHook


class CDAN(GAN):
    """
    Wraps [CDANHook][pytorch_adapt.hooks.CDANHook].

    |Container|Required keys|
    |---|---|
    |models|```["G", "C", "D"]```|
    |optimizers|```["G", "C", "D"]```|
    |misc|```["feature_combiner"]```|
    """

    @property
    def hook_cls(self):
        return CDANHook

    def get_key_enforcer(self) -> KeyEnforcer:
        ke = super().get_key_enforcer()
        ke.requirements["misc"] = ["feature_combiner"]
        return ke


class CDANE(CDAN):
    @property
    def hook_cls(self):
        return CDANEHook


class DomainConfusion(GAN):
    """
    Wraps [DomainConfusionHook][pytorch_adapt.hooks.DomainConfusionHook].

    |Container|Required keys|
    |---|---|
    |models|```["G", "C", "D"]```|
    |optimizers|```["G", "C", "D"]```|
    """

    @property
    def hook_cls(self):
        return DomainConfusionHook


class VADA(GAN):
    """
    Wraps [VADAHook][pytorch_adapt.hooks.VADAHook].

    |Container|Required keys|
    |---|---|
    |models|```["G", "C", "D"]```|
    |optimizers|```["G", "C", "D"]```|
    |misc|```["combined_model"]```|

    The ```"combined_model"``` key does not need to be passed in.
    It is simply ```torch.nn.Sequential(G, C)```, and is set
    automatically during initialization.
    """

    @property
    def hook_cls(self):
        return VADAHook

    def init_containers_and_check_keys(self, containers):
        models = containers["models"]
        misc = containers["misc"]
        misc["combined_model"] = torch.nn.Sequential(models["G"], models["C"])
        super().init_containers_and_check_keys(containers)

    def get_key_enforcer(self) -> KeyEnforcer:
        ke = super().get_key_enforcer()
        ke.requirements["misc"] = ["combined_model"]
        return ke
